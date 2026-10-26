# -*- coding: utf-8 -*-
"""
情景记忆 · 转生版（Episodic Memory with Rebirth）
────────────────────────────────────────────────────────────
四层记忆：
  L0 热上下文   最近对话（AstrBot 自己的会话历史）
  L1 灵魂       核心记忆 + 大事记 + 最近卡片摘要（新会话注入）
  L2 记忆卡片   summary + metadata + embedding（检索入口）
  L3 原始细节   facts（按 card_id 外挂，不向量化）

检索：metadata pre-filter → 余弦相似度 → 三因子打分
      语义 0.7 + 时间新鲜度 0.2 + 强化度 0.1

工具：
  memory_note    手动记一张卡片（或追加事实到已有卡片）
  memory_recall  检索记忆卡片
  memory_detail  展开某张卡片的细节
  memory_forget  删除卡片
"""

import asyncio
import json
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

import aiohttp

import astrbot.api.event.filter as filter
from astrbot.api import llm_tool, logger, star
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

# ── 默认参数（可被插件配置覆盖）──────────────────────────────
DEFAULTS = {
    "embed_url": "https://open.bigmodel.cn/api/paas/v4/embeddings",
    "embed_key": "",
    "embed_model": "embedding-3",
    "session_gap_minutes": 30,
    "extract_every_n": 4,
    "extract_enabled": True,
    "recall_top_k": 5,
    "recall_min_score": 0.18,
    "w_semantic": 0.7,
    "w_time": 0.2,
    "w_hit": 0.1,
    "time_decay_days": 90,
    "soul_recent_cards": 5,
    "max_facts_return": 20,
    "context_token_limit": 32000,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    entities    TEXT DEFAULT '[]',
    keywords    TEXT DEFAULT '[]',
    topics      TEXT DEFAULT '[]',
    ts          INTEGER NOT NULL,
    embedding   BLOB,
    hit_count   INTEGER DEFAULT 0,
    last_used   INTEGER DEFAULT 0,
    archived    INTEGER DEFAULT 0,
    important   INTEGER DEFAULT 0,
    scope       TEXT DEFAULT 'private',
    owner_id    TEXT DEFAULT '',
    owner_name  TEXT DEFAULT '',
    type        TEXT DEFAULT 'fact',
    confidence  REAL DEFAULT 0.8
);
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    ts          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_card ON facts(card_id);
CREATE INDEX IF NOT EXISTS idx_cards_ts ON cards(ts);
"""


def _bigrams(text: str) -> set[str]:
    if not text:
        return set()
    t = re.sub(r"\s+", "", text)
    grams = {t[i : i + 2] for i in range(len(t) - 1)}
    for w in re.findall(r"[a-zA-Z0-9_]{2,}", text):
        grams.add(w.lower())
    return grams


class EpisodicMemoryPlugin(star.Star):
    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        self.context = context
        self.config = config or {}
        # [规范] 数据必须放在 data/plugin_data/<plugin_name>，便于迁移和审计
        self.data_dir = Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_plugin_episodic_memory"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memory.db"
        self.soul_file = self.data_dir / "soul.md"
        self.soul_file.touch(exist_ok=True)
        self._cfg = dict(DEFAULTS)
        self._sessions: dict[str, dict] = {}
        self._tasks: set = set()   # 持有后台任务引用，防止被 GC 回收
        self._rerank_cache: dict = {}   # 重排结果缓存（QPS 低，简单 LRU 即可）
        self._init_db()

    async def initialize(self):
        """启动预热：提前加载重排模型，避免第一条消息卡 2~3 秒。"""
        try:
            await asyncio.to_thread(self._get_reranker)
            logger.info("[episodic] 重排模型预热完成")
        except Exception as e:
            logger.warning(f"[episodic] 重排模型预热失败: {e}")

    # ── 配置 ────────────────────────────────────────────────
    def _cfg_get(self, key):
        try:
            c = self.config or {}
            if key in c and c[key] not in ("", None):
                return c[key]
        except Exception:
            pass
        return self._cfg.get(key)

    # ── 数据库 ──────────────────────────────────────────────
    def _conn(self):
        c = sqlite3.connect(str(self.db_path), timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = [r[1] for r in c.execute("PRAGMA table_info(cards)")]
            if "scope" not in cols:
                c.execute("ALTER TABLE cards ADD COLUMN scope TEXT DEFAULT 'private'")
            if "owner_id" not in cols:
                c.execute("ALTER TABLE cards ADD COLUMN owner_id TEXT DEFAULT ''")
            if "owner_name" not in cols:
                c.execute("ALTER TABLE cards ADD COLUMN owner_name TEXT DEFAULT ''")
            if "type" not in cols:
                c.execute("ALTER TABLE cards ADD COLUMN type TEXT DEFAULT 'fact'")
            if "confidence" not in cols:
                c.execute("ALTER TABLE cards ADD COLUMN confidence REAL DEFAULT 0.8")

    def _maybe_archive(self):
        """[遗忘机制] 90 天未用 + 命中 < 3 的卡片归档（不删，只是不再被检索）。"""
        cutoff = int((time.time() - 90 * 86400) * 1000)
        with self._conn() as c:
            n = c.execute(
                "UPDATE cards SET archived=1 "
                "WHERE archived=0 AND COALESCE(important,0)=0 AND ts < ? AND COALESCE(hit_count,0) < 3",
                (cutoff,),
            ).rowcount
        if n:
            logger.info(f"[episodic] 自动归档 {n} 张老旧卡片（遗忘，非删除）")

    def _admin_ids(self) -> set[str]:
        """AstrBot 配置里的管理员 QQ 集合。"""
        try:
            cfg = self.context.get_config() or {}
            return {str(x).strip() for x in (cfg.get("admins_id") or []) if str(x).strip()}
        except Exception:
            return set()

    def _can_modify(self, event, card_owner: str) -> bool:
        """能否修改某张卡片：自己的可以；管理员可以改任何人；非管理员不能改管理员的。"""
        sender = str(getattr(event, "get_sender_id", lambda: "")() or "").strip()
        owner = str(card_owner or "").strip()
        if self._is_owner(event):
            return True
        if owner in self._admin_ids():
            return False
        return owner == sender

    _TYPE_KEYWORDS = {
        "rule": ["以后", "不要", "必须", "记住", "别忘了", "你得", "禁止", "只能", "不准", "不许"],
        "preference": ["喜欢", "爱", "讨厌", "不喜欢", "偏好", "习惯"],
        "emotion": ["烦", "开心", "难过", "生气", "爽", "累", "崩溃"],
    }

    def _classify_type(self, summary: str, llm_type: str = "") -> str:
        """三层分类：LLM 非 fact 就信它；否则用关键词兜底。"""
        t = (llm_type or "").strip().lower()
        if t in ("preference", "rule", "emotion"):
            return t
        text = summary or ""
        for kw_type, kws in self._TYPE_KEYWORDS.items():
            if any(k in text for k in kws):
                return kw_type
        return "fact"

    _CHATTER = re.compile(
        r"^[\s哈呵嘿嗯哦噢啊呀吧啦呢了噢喔耶…。，！？、~～\.\,\!\?\s]+$"
        r"|^(在吗|在不在|好的?|嗯+|哈哈+|草+|牛|6|666|？{2,}|。{3,})$"
    )

    def _worth_extracting(self, turns) -> bool:
        """过滤层：纯闲聊 / 太短 / 无信息量 → 不提炼，省 API。"""
        user_texts = [str(t.get("text") or "").strip() for t in turns if t.get("role") == "user"]
        text = " ".join(user_texts).strip()
        if len(text) < 8:
            return False
        if self._CHATTER.match(text):
            return False
        if not re.search(r"\d|[A-Za-z]{3,}|[\u4e00-\u9fa5]{4,}", text):
            return False
        return True

    _CLOSING = re.compile(
        r"^(好的?|好|行|可以|谢谢|谢|嗯+|ok|okay|懂了|明白|知道了|收到|就这样|没事了|行吧|好嘞|多谢|辛苦了)$",
        re.I,
    )

    def _topic_closed(self, turns) -> bool:
        """触发层·话题终结：最近 3 条用户消息里至少 2 条是收尾语或短句。"""
        users = [str(t.get("text") or "").strip() for t in turns if t.get("role") == "user"]
        if len(users) < 2:
            return False
        last3 = users[-3:]
        closing = sum(1 for u in last3 if self._CLOSING.match(u) or len(u) <= 4)
        return closing >= 2

    async def _idle_extract(self, umo: str, gap: float = 300):
        """触发层·会话结束：gap 秒无新消息 → 触发提炼。"""
        try:
            await asyncio.sleep(gap)
        except asyncio.CancelledError:
            return
        st = self._st(umo)
        turns = st["pending"][:]
        if not turns:
            return
        st["pending"] = []
        if not self._worth_extracting(turns):
            logger.debug("[episodic] 会话结束触发，但过滤层判定无信息量")
            return
        logger.info("[episodic] 会话结束触发提炼（5 分钟无消息）")
        await self._extract(
            umo,
            turns,
            st.get("last_scope") or "shared",
            st.get("last_owner_id") or "",
            st.get("last_owner_name") or "",
        )

    def _sanitize_entities(self, entities, sender_id: str) -> list:
        """[防伪装] 实体里不允许出现别人的 QQ 号，避免 A 把自己记成 B。"""
        sender = str(sender_id or "").strip()
        out = []
        for e in entities or []:
            e = str(e).strip()
            if not e:
                continue
            ids = re.findall(r"\d{6,12}", e)
            # 出现任何不是自己的 QQ 号 → 丢弃这个实体
            if ids and any(i != sender for i in ids):
                logger.warning(f"[episodic] 拦截疑似伪装实体: {e} (sender={sender})")
                continue
            out.append(e)
        return out[:10]

    @staticmethod
    def _is_owner(event) -> bool:
        try:
            return str(getattr(event, "role", "")).strip().lower() == "admin"
        except Exception:
            return False

    @staticmethod
    def _session_scope(event_or_umo) -> str:
        """优先用消息类型判断，退化时再按 umo 字符串猜。"""
        try:
            mt = getattr(event_or_umo, "get_message_type", None)
            if callable(mt):
                return "shared" if "group" in str(mt()).lower() else "private"
        except Exception:
            pass
        return "shared" if "GroupMessage" in str(event_or_umo) else "private"

    # ── 灵魂（核心记忆 + 大事记）─────────────────────────────
    def _read_soul(self) -> str:
        try:
            return self.soul_file.read_text(encoding="utf-8")[:3072]
        except Exception:
            return ""

    def _write_soul(self, text: str):
        self.soul_file.write_text(str(text or "")[:3072], encoding="utf-8")

    # ── 向量 ────────────────────────────────────────────────
    async def _embed(self, text: str):
        key = str(self._cfg_get("embed_key") or "").strip()
        if not key:
            return None
        url = str(self._cfg_get("embed_url"))
        model = str(self._cfg_get("embed_model"))
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "input": text[:2000]},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status != 200:
                        logger.warning(f"[episodic] embedding HTTP {r.status}")
                        return None
                    j = await r.json()
                    return j["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"[episodic] embedding 失败: {e}")
            return None

    @staticmethod
    def _cos(a, b) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(n))
        na = math.sqrt(sum(x * x for x in a[:n]))
        nb = math.sqrt(sum(x * x for x in b[:n]))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _pack(vec):
        return json.dumps(vec).encode("utf-8") if vec else None

    @staticmethod
    def _unpack(blob):
        if not blob:
            return None
        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return None

    # ── 卡片写入 ────────────────────────────────────────────
    async def _upsert_card(
        self,
        summary: str,
        entities=None,
        keywords=None,
        topics=None,
        facts=None,
        card_id: str | None = None,
        important: bool = False,
        scope: str = "private",
        owner_id: str = "",
        owner_name: str = "",
        card_type: str = "fact",
        confidence: float = 0.8,
    ) -> str:
        now = int(time.time() * 1000)
        vec = await self._embed(summary)

        # [归档层] 向量去重：相似度 >= 0.85 → 合并到旧卡片，防止记忆膨胀
        if not card_id and vec:
            best_id, best_sim = None, 0.0
            with self._conn() as c:
                rows = c.execute("SELECT id, embedding FROM cards WHERE archived=0").fetchall()
            for r in rows:
                ov = self._unpack(r["embedding"])
                if ov:
                    s = self._cos(vec, ov)
                    if s > best_sim:
                        best_sim, best_id = s, r["id"]
            if best_id and best_sim >= 0.85:
                logger.info(f"[episodic] 向量去重命中 {best_sim:.2f} → 合并到 {best_id}")
                card_id = best_id

        with self._conn() as c:
            if card_id:
                row = c.execute("SELECT id FROM cards WHERE id=?", (card_id,)).fetchone()
                if row:
                    c.execute(
                        "UPDATE cards SET summary=?, entities=?, keywords=?, topics=?, embedding=?, "
                        "scope=?, important=MAX(COALESCE(important,0), ?), type=?, confidence=? WHERE id=?",
                        (
                            summary,
                            json.dumps(entities or [], ensure_ascii=False),
                            json.dumps(keywords or [], ensure_ascii=False),
                            json.dumps(topics or [], ensure_ascii=False),
                            self._pack(vec),
                            scope,
                            1 if important else 0,
                            card_type,
                            confidence,
                            card_id,
                        ),
                    )
                    for f in facts or []:
                        c.execute(
                            "INSERT INTO facts(card_id, text, ts) VALUES(?,?,?)",
                            (card_id, str(f)[:1000], now),
                        )
                    return card_id
            cid = card_id or ("card_" + uuid.uuid4().hex[:12])
            c.execute(
                "INSERT INTO cards(id,summary,entities,keywords,topics,ts,embedding,hit_count,last_used,archived,important,scope,owner_id,owner_name,type,confidence) "
                "VALUES(?,?,?,?,?,?,?,0,0,0,?,?,?,?,?,?)",
                (
                    cid,
                    summary,
                    json.dumps(entities or [], ensure_ascii=False),
                    json.dumps(keywords or [], ensure_ascii=False),
                    json.dumps(topics or [], ensure_ascii=False),
                    now,
                    self._pack(vec),
                    1 if important else 0,
                    scope,
                    owner_id,
                    owner_name,
                    card_type,
                    confidence,
                ),
            )
            for f in facts or []:
                c.execute(
                    "INSERT INTO facts(card_id, text, ts) VALUES(?,?,?)",
                    (cid, str(f)[:1000], now),
                )
        return cid

    # ── 检索 ────────────────────────────────────────────────
    def _score(self, sem: float, ts: int, hit: int) -> float:
        now = time.time()
        days = max(0.0, (now - ts / 1000) / 86400)
        decay = float(self._cfg_get("time_decay_days") or 90)
        fresh = math.exp(-days / max(1.0, decay))
        reinf = min(1.0, math.log1p(max(0, hit)) / math.log(20))
        return (
            float(self._cfg_get("w_semantic")) * sem
            + float(self._cfg_get("w_time")) * fresh
            + float(self._cfg_get("w_hit")) * reinf
        )

    async def _recall(self, query: str, top_k: int | None = None, owner: bool = False):
        query = (query or "").strip()
        if not query:
            return []
        top_k = int(top_k or self._cfg_get("recall_top_k") or 5)
        qvec = await self._embed(query)
        qgrams = _bigrams(query)

        with self._conn() as c:
            if owner:
                rows = c.execute(
                    "SELECT * FROM cards WHERE archived=0 ORDER BY ts DESC LIMIT 2000"
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM cards WHERE archived=0 AND scope='shared' ORDER BY ts DESC LIMIT 2000"
                ).fetchall()

        candidates = []
        for r in rows:
            entities = json.loads(r["entities"] or "[]")
            keywords = json.loads(r["keywords"] or "[]")
            text = (r["summary"] or "") + " " + " ".join(entities + keywords)
            grams = _bigrams(text)
            overlap = len(qgrams & grams) / max(1, len(qgrams)) if qgrams else 0
            sem = self._cos(qvec, self._unpack(r["embedding"])) if qvec else overlap
            # 预过滤：实体/关键词命中（含二元组粗匹配）或 二元组重合 或 语义足够高
            hit_tag = False
            for t in entities + keywords:
                if not t:
                    continue
                if t in query or (qgrams and (_bigrams(t) & qgrams)):
                    hit_tag = True
                    break
            if not hit_tag and overlap < 0.12 and sem < 0.35:
                continue
            s = self._score(sem, r["ts"], r["hit_count"] or 0)
            candidates.append((s, r))

        candidates.sort(key=lambda x: x[0], reverse=True)
        picked = candidates[:top_k]
        # 记录命中 + 强化
        if picked:
            now = int(time.time() * 1000)
            with self._conn() as c:
                for _, r in picked:
                    c.execute(
                        "UPDATE cards SET hit_count=hit_count+1, last_used=? WHERE id=?",
                        (now, r["id"]),
                    )
        return [
            {
                "id": r["id"],
                "summary": r["summary"],
                "entities": json.loads(r["entities"] or "[]"),
                "keywords": json.loads(r["keywords"] or "[]"),
                "topics": json.loads(r["topics"] or "[]"),
                "ts": r["ts"],
                "score": round(s, 3),
            }
            for s, r in picked
        ]

    # ── 重排（bge-reranker，本地跑，不耗 API token）──────────
    _rerank_model = None

    def _get_reranker(self):
        if self._rerank_model is None:
            try:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                from sentence_transformers import CrossEncoder

                self._rerank_model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
                logger.info("[episodic] 重排模型 bge-reranker-base 已加载")
            except Exception as e:
                logger.warning(f"[episodic] 重排模型加载失败（降级为不重排）: {e}")
                self._rerank_model = False
        return self._rerank_model or None

    async def _rerank(self, query: str, cards: list, top_k: int = 3) -> list:
        """重排：用 bge-reranker 对候选卡片二次排序，只留最相关的 top_k 条。"""
        m = self._get_reranker()
        if not m or len(cards) <= 1:
            return cards[:top_k]

        # 简单 LRU：同一 (query, 卡片集合) 直接返回缓存结果
        key = (query, tuple(c["id"] for c in cards))
        cached = self._rerank_cache.get(key)
        if cached is not None:
            return cached

        try:
            pairs = [(query, c["summary"]) for c in cards]
            scores = await asyncio.wait_for(
                asyncio.to_thread(m.predict, pairs), timeout=2.0
            )
            ranked = sorted(zip(cards, scores), key=lambda x: -float(x[1]))
            kept = [c for c, s in ranked[:top_k] if float(s) > 0.01]
            out = kept or cards[:top_k]
        except asyncio.TimeoutError:
            logger.warning("[episodic] 重排超时（>2s），降级用原顺序")
            return cards[:top_k]
        except Exception as e:
            logger.warning(f"[episodic] 重排失败（降级为原顺序）: {e}")
            return cards[:top_k]

        if len(self._rerank_cache) >= 200:
            self._rerank_cache.clear()
        self._rerank_cache[key] = out
        return out

    def _facts(self, card_id: str, limit: int | None = None):
        limit = int(limit or self._cfg_get("max_facts_return") or 20)
        with self._conn() as c:
            rows = c.execute(
                "SELECT text, ts FROM facts WHERE card_id=? ORDER BY ts DESC LIMIT ?",
                (card_id, limit),
            ).fetchall()
        return [{"text": r["text"], "ts": r["ts"]} for r in rows]

    # ── 会话状态 ────────────────────────────────────────────
    def _st(self, umo: str) -> dict:
        st = self._sessions.get(umo)
        if st is None:
            st = {"last_ts": 0, "pending": [], "soul_injected": False, "idle_task": None}
            self._sessions[umo] = st
        return st

    # ── 提炼卡片 ────────────────────────────────────────────
    async def _extract(
        self,
        umo: str,
        turns: list,
        scope: str = "private",
        owner_id: str = "",
        owner_name: str = "",
    ):
        if not turns:
            return
        provider = await self.context.get_using_provider_async(umo)
        if not provider:
            return
        def _fmt_turn(t):
            if t.get("role") == "user":
                return f"- {t.get('name') or '某人'}（QQ {t.get('qq') or '?'}）: {t['text']}"
            return f"- 你（机器人）: {t['text']}"

        convo = "\n".join(_fmt_turn(t) for t in turns)
        prompt = (
            "请从以下对话中提取有价值的长期记忆，输出 JSON。\n"
            "每条卡片用 type 区分：fact=客观事实, preference=偏好, rule=行为规则, emotion=情绪状态。\n"
            "没有值得记的就返回 {\"cards\":[]}。\n\n"
            f"【对话】\n{convo[:3000]}\n\n"
            '{"cards":[{"summary":"≤80字","type":"fact","confidence":0.9,'
            '"entities":["昵称（QQ xxx）"],"keywords":["..."],"topics":["..."],"facts":["..."]}]}\n'
            "硬性要求：\n"
            "1. entities 抄对话里的「昵称（QQ xxx）」，不要编、不要合并；\n"
            "2. 同名不同 QQ 是不同的人，必须分开记；\n"
            "3. 禁止用「用户/管理员/你们/大家」当实体；\n"
            "4. 只记事实/偏好/规则/情绪，不记寒暄；\n"
            "5. confidence 0~1，越确定越高；\n"
            "6. 涉及「关系约定 / 底线 / 重要决策」的卡片，必须标 important=true。\n"
            "只输出 JSON。"
        )
        try:
            resp = await provider.text_chat(prompt=prompt, session_id=f"extract-{uuid.uuid4().hex[:8]}")
            text = (getattr(resp, "completion_text", "") or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            # 容错：只截取第一个 { 到最后一个 } 之间，忽略前后解释文字
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise ValueError("回复里没有 JSON")
            data = json.loads(m.group(0))
            for card in data.get("cards", [])[:2]:
                await self._upsert_card(
                    summary=str(card.get("summary", ""))[:500],
                    entities=self._sanitize_entities(card.get("entities") or [], owner_id),
                    keywords=card.get("keywords") or [],
                    topics=card.get("topics") or [],
                    facts=card.get("facts") or [],
                    scope=scope,
                    owner_id=owner_id,
                    owner_name=owner_name,
                    card_type=self._classify_type(
                        str(card.get("summary") or ""),
                        str(card.get("type") or ""),
                    ) if isinstance(card, dict) else "fact",
                    confidence=float(card.get("confidence") or 0.8) if isinstance(card, dict) else 0.8,
                    important=bool(card.get("important")) if isinstance(card, dict) else False,
                )
            logger.info(f"[episodic] 提炼 {len(data.get('cards', []))} 张卡片")
            await asyncio.to_thread(self._maybe_archive)
        except Exception as e:
            logger.warning(f"[episodic] 提炼失败: {e}")

    # ── 注入 ────────────────────────────────────────────────
    def _soul_block(self, recent_cards: list, include_soul: bool = True) -> str:
        parts = []
        if include_soul:
            soul = self._read_soul().strip()
            if soul:
                parts.append("【灵魂 / 核心记忆（必须遵守）】\n" + soul)
        if recent_cards:
            lines = []
            for r in recent_cards:
                ts = r["ts"]
                try:
                    day = time.strftime("%m-%d", time.localtime(ts / 1000))
                except Exception:
                    day = "??-??"
                lines.append(f"- ({day}) {r['summary']}")
            parts.append("【上次聊到（最近记得的事）】\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _recent_cards(self, n: int = 5, owner: bool = False) -> list:
        with self._conn() as c:
            if owner:
                rows = c.execute(
                    "SELECT id,summary,ts FROM cards WHERE archived=0 ORDER BY ts DESC LIMIT ?",
                    (int(n),),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id,summary,ts FROM cards WHERE archived=0 AND scope='shared' ORDER BY ts DESC LIMIT ?",
                    (int(n),),
                ).fetchall()
        return [{"id": r["id"], "summary": r["summary"], "ts": r["ts"]} for r in rows]

    @filter.on_llm_request(priority=100)
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            umo = event.unified_msg_origin
            st = self._st(umo)
            now = time.time()
            gap = float(self._cfg_get("session_gap_minutes") or 30) * 60
            owner = self._is_owner(event)

            # /reset 或新对话检测：conversation_id 变了就重注入灵魂
            try:
                conv_id = await self.context.conversation_manager.get_curr_conversation_id(umo)
            except Exception:
                conv_id = None
            if conv_id is not None:
                if st.get("conv_id") not in (None, conv_id):
                    st["soul_injected"] = False
                    st["last_ts"] = 0
                    st["pending"] = []
                st["conv_id"] = conv_id

            # 新会话（超时或 /reset）→ 注入灵魂
            if now - st["last_ts"] > gap or not st["soul_injected"]:
                recent = self._recent_cards(
                    int(self._cfg_get("soul_recent_cards") or 5),
                    owner=owner,
                )
                block = self._soul_block(recent, include_soul=owner)
                if block:
                    # [规范] 不污染 system_prompt，改用临时用户内容（不影响 prompt 缓存）
                    req.extra_user_content_parts.append(
                        TextPart(text="\n\n" + block + "\n\n").make_as_temp()
                    )
                st["soul_injected"] = True
                st["pending"] = []

            # 按需召回
            query = (event.message_str or "").strip()
            if query:
                recalled = await self._recall(query, top_k=10, owner=owner)
                if recalled:
                    # [重排] 向量 Top-10 → bge-reranker 二次排序 → Top-3
                    recalled = await self._rerank(query, recalled, top_k=3)
                    # [注入预算] 硬上限 2000 字符，超了按得分丢尾部
                    budget = 2000
                    used = 0
                    lines = []
                    for r in recalled:
                        ts = r["ts"]
                        try:
                            day = time.strftime("%m-%d", time.localtime(ts / 1000))
                        except Exception:
                            day = "??-??"
                        line = f"- [{r['id']}] ({day}) {r['summary']}"
                        if used + len(line) > budget:
                            continue
                        lines.append(line)
                        used += len(line)
                    if lines:
                        # [规范] 召回结果作为临时用户内容注入，system_prompt 保持不变
                        req.extra_user_content_parts.append(
                            TextPart(
                                text="\n\n【长期记忆档案】（背景参考；若与当前对话冲突，以当前对话为准）\n"
                                + "\n".join(lines)
                                + "\n\n"
                            ).make_as_temp()
                        )

            st["last_ts"] = now
        except Exception as e:
            logger.error(f"[episodic] 注入失败: {e}")

    @filter.on_llm_response()
    async def collect_turn(self, event: AstrMessageEvent, response: LLMResponse):
        try:
            umo = event.unified_msg_origin
            st = self._st(umo)
            user_text = (event.message_str or "").strip()
            bot_text = (getattr(response, "completion_text", "") or "").strip()
            if user_text:
                st["pending"].append({
                    "role": "user",
                    "text": user_text[:500],
                    "name": str(getattr(event, "get_sender_name", lambda: "")() or ""),
                    "qq": str(getattr(event, "get_sender_id", lambda: "")() or ""),
                })
            if bot_text:
                st["pending"].append({"role": "assistant", "text": bot_text[:500]})

            every_n = int(self._cfg_get("extract_every_n") or 4)
            # [触发层] 显式指令 / 攒够 N 轮（兜底）
            explicit = any(
                k in (event.message_str or "")
                for k in ("记住", "记一下", "以后都", "以后按", "别忘了", "你得知道")
            )
            closed = self._topic_closed(st["pending"])

            # 记录身份，供「会话结束」触发时使用
            st["last_owner_id"] = str(getattr(event, "get_sender_id", lambda: "")() or "")
            st["last_owner_name"] = str(getattr(event, "get_sender_name", lambda: "")() or "")
            st["last_scope"] = self._session_scope(event)

            # [触发层] 5 分钟无新消息 → 会话结束提炼（每条新消息都重置计时）
            old = st.get("idle_task")
            if old:
                old.cancel()
            idle = asyncio.create_task(self._idle_extract(umo, 300))
            st["idle_task"] = idle
            self._tasks.add(idle)
            idle.add_done_callback(self._tasks.discard)

            if self._cfg_get("extract_enabled") and (explicit or closed or len(st["pending"]) >= every_n * 2):
                turns = st["pending"][:]
                st["pending"] = []
                if not self._worth_extracting(turns):
                    logger.debug("[episodic] 过滤层拦截：纯闲聊 / 无信息量，不消耗提炼 API")
                else:
                    task = asyncio.create_task(
                        self._extract(
                            umo,
                            turns,
                            self._session_scope(event),
                            str(getattr(event, "get_sender_id", lambda: "")() or ""),
                            str(getattr(event, "get_sender_name", lambda: "")() or ""),
                        )
                    )
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

            # [新增] 上下文 token 超阈值 → 自动重置（记忆卡片保留，转生重新注入）
            try:
                limit = int(self._cfg_get("context_token_limit") or 32000)
                if limit > 0:
                    cm = self.context.conversation_manager
                    cid = await cm.get_curr_conversation_id(umo)
                    if cid:
                        conv = await cm.get_conversation(umo, cid)
                        usage = int(getattr(conv, "token_usage", 0) or 0) if conv else 0
                        if usage >= limit:
                            # [兜底] 转生前，把 pending 里还没提炼的再提炼一次（避免丢记忆）
                            leftovers = st["pending"][:]
                            if leftovers and self._worth_extracting(leftovers):
                                try:
                                    await self._extract(
                                        umo,
                                        leftovers,
                                        self._session_scope(event),
                                        str(getattr(event, "get_sender_id", lambda: "")() or ""),
                                        str(getattr(event, "get_sender_name", lambda: "")() or ""),
                                    )
                                    logger.info("[episodic] 转生前兜底提炼完成")
                                except Exception as e:
                                    logger.warning(f"[episodic] 转生前兜底提炼失败: {e}")
                            # 尽量保留原会话的人格绑定
                            old_persona = (
                                getattr(conv, "persona_id", None) if conv else None
                            )
                            await cm.new_conversation(umo, persona_id=old_persona)
                            st["soul_injected"] = False
                            st["last_ts"] = 0
                            st["conv_id"] = None
                            st["pending"] = []
                            logger.info(
                                f"[episodic] 上下文 {usage} token 超过 {limit}，已自动重置并准备转生"
                            )
            except Exception as e:
                logger.warning(f"[episodic] token 阈值检查失败: {e}")
        except Exception as e:
            logger.error(f"[episodic] 收集失败: {e}")

    # ── 工具 ────────────────────────────────────────────────
    @llm_tool(name="memory_note")
    async def memory_note(
        self,
        event: AstrMessageEvent,
        summary: str,
        entities: str = "",
        keywords: str = "",
        topics: str = "",
        facts: str = "",
        important: bool = False,
        scope: str = "private",
    ) -> str:
        """把一件事记成记忆卡片（或追加到已有卡片）。

        Args:
            summary(string): 一句话摘要，≤80字，保留情绪和最新进展
            entities(string): 实体，逗号分隔，**必须写具体的人**：昵称/QQ号/组织名，如「东/1705007522,父亲,老张」；禁止用「用户」「管理员」「你们」这类泛指
            keywords(string): 关键词，逗号分隔，如「裁员,失业,赔偿金」
            topics(string): 话题，逗号分隔，如「家庭,工作」
            facts(string): 具体细节，逗号分隔，如「赔偿金没谈拢,经济压力大」
            important(bool): 是否重要记忆（永不压缩）
            scope(string): private=只给管理员看；shared=群里/朋友也能看到
        """
        def _split(s):
            return [x.strip() for x in str(s or "").split(",") if x.strip()]

        # [隐私] 非管理员只能写 shared，不能创建 private 卡片
        if not self._is_owner(event):
            scope = "shared"

        # 尝试找同一话题的卡片原地更新（summary 相似 + 实体有交集 + 只能改自己的）
        target = None
        hit = _bigrams(summary)
        new_ents = set(_split(entities))
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,summary,entities,owner_id FROM cards WHERE archived=0 ORDER BY ts DESC LIMIT 500"
            ).fetchall()
        best = 0.0
        for r in rows:
            # 所有权：只能改自己的卡片（管理员例外，但不能改管理员的）
            if not self._can_modify(event, r["owner_id"]):
                continue
            r_ents = set(json.loads(r["entities"] or "[]"))
            if new_ents and r_ents and not (new_ents & r_ents):
                continue
            s = len(hit & _bigrams(r["summary"])) / max(1, len(hit))
            if s > 0.65 and s > best:
                best, target = s, r["id"]

        sender = str(getattr(event, "get_sender_id", lambda: "")() or "").strip()
        sender_name = str(getattr(event, "get_sender_name", lambda: "")() or "").strip()

        cid = await self._upsert_card(
            summary=summary,
            entities=self._sanitize_entities(_split(entities), sender),
            keywords=_split(keywords),
            topics=_split(topics),
            facts=_split(facts),
            card_id=target,
            important=important,
            scope=("shared" if str(scope).strip() == "shared" else "private"),
            owner_id=sender,
            owner_name=sender_name,
        )
        return f"记住了（卡片 {cid}，scope={scope}）"

    @llm_tool(name="memory_recall")
    async def memory_recall(self, event: AstrMessageEvent, query: str, limit: int = 6) -> str:
        """检索过去的记忆卡片。

        Args:
            query(string): 搜索词，比如「父亲工作」「那个赔偿金」
            limit(number): 最多几条，默认 6，最大 20
        """
        try:
            n = max(1, min(20, int(limit)))
        except Exception:
            n = 6
        items = await self._recall(query, n, owner=self._is_owner(event))
        if not items:
            return "（没找到相关记忆）"
        lines = []
        for r in items:
            day = time.strftime("%m-%d", time.localtime(r["ts"] / 1000))
            lines.append(f"- [{r['id']}] ({day}) {r['summary']}")
        return "\n".join(lines)

    @llm_tool(name="memory_detail")
    async def memory_detail(self, event: AstrMessageEvent, card_id: str) -> str:
        """展开某张记忆卡片的全部细节。

        Args:
            card_id(string): 卡片 id，来自 memory_recall 的结果
        """
        with self._conn() as c:
            row = c.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
        if not row:
            return f"（没有卡片 {card_id}）"
        # [隐私] 非管理员不能看私密卡片
        if not self._is_owner(event) and str(row["scope"] or "private") != "shared":
            return "⚠️ 这条记忆是私密的，只有管理员能看。"
        facts = self._facts(card_id)
        lines = [
            f"摘要：{row['summary']}",
            f"归属：{row['owner_name'] or '未知'}（{row['owner_id'] or '?'}）",
            f"实体：{row['entities']}",
            f"关键词：{row['keywords']}",
            f"话题：{row['topics']}",
        ]
        if facts:
            lines.append("细节：")
            for f in facts:
                day = time.strftime("%m-%d", time.localtime(f["ts"] / 1000))
                lines.append(f"  - ({day}) {f['text']}")
        else:
            lines.append("细节：（暂无）")
        return "\n".join(lines)

    @llm_tool(name="memory_forget")
    async def memory_forget(self, event: AstrMessageEvent, card_id: str) -> str:
        """删除一张记忆卡片（记错了才用）。

        Args:
            card_id(string): 卡片 id
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT summary, owner_id FROM cards WHERE id=?", (card_id,)
            ).fetchone()
        if not row:
            return f"（没有卡片 {card_id}）"
        # 所有权：只能删自己的卡片（管理员例外，但不能删管理员的）
        if not self._can_modify(event, row["owner_id"]):
            return "⚠️ 这张卡片不属于你，不能删。"
        with self._conn() as c:
            c.execute("DELETE FROM facts WHERE card_id=?", (card_id,))
            c.execute("DELETE FROM cards WHERE id=?", (card_id,))
        return f"已删除：{row['summary'][:50]}"

    @llm_tool(name="memory_soul")
    async def memory_soul(self, event: AstrMessageEvent, content: str = "") -> str:
        """读取或重写灵魂记忆（核心记忆：她是谁/你是谁/底线/大事记）。

        Args:
            content(string): 新的灵魂记忆全文；留空则读取
        """
        # [隐私] 灵魂记忆只有管理员能读写
        if not self._is_owner(event):
            return "⚠️ 灵魂记忆只有管理员能读写。"
        if content.strip():
            self._write_soul(content)
            return "灵魂记忆已重写（≤3KB）"
        return self._read_soul() or "（灵魂记忆为空）"
