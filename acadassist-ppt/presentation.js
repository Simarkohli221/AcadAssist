const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaBrain, FaSearch, FaDatabase, FaRobot, FaChartBar,
        FaCheckCircle, FaExclamationTriangle, FaLayerGroup,
        FaFileAlt, FaCogs, FaLightbulb, FaCode, FaMicrochip,
        FaArrowRight, FaGraduationCap, FaUniversity, FaFlask,
        FaSyncAlt, FaFilter, FaListAlt, FaCloud, FaShieldAlt } = require("react-icons/fa");
const { MdOutlineSchema, MdAutoFixHigh } = require("react-icons/md");

// ─── Color Palette ─────────────────────────────────────────────────────────────
const C = {
  navy:     "0D1B2A",   // dominant dark
  indigo:   "1B3A6B",   // secondary
  violet:   "7B2FBE",   // accent
  vivid:    "A855F7",   // bright accent
  teal:     "0891B2",   // alt accent
  white:    "FFFFFF",
  offwhite: "F0F4FF",
  silver:   "CBD5E1",
  muted:    "94A3B8",
  dark:     "0A0F1E",
  card:     "132044",   // card bg
  cardL:    "1E3A6E",   // card light bg
  green:    "10B981",
  amber:    "F59E0B",
  red:      "EF4444",
};

// ─── Icon helper ───────────────────────────────────────────────────────────────
async function ico(IconComp, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComp, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ─── Helpers ───────────────────────────────────────────────────────────────────
function darkSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: C.navy };
  return s;
}
function lightSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  return s;
}
function slideTitle(s, text, y = 0.28, color = C.white, size = 30) {
  s.addText(text, {
    x: 0.5, y, w: 9, h: 0.55,
    fontSize: size, bold: true, color, fontFace: "Calibri",
    align: "left", margin: 0
  });
}
function slideTitleDark(s, text) {
  s.addText(text, {
    x: 0.5, y: 0.28, w: 9, h: 0.55,
    fontSize: 30, bold: true, color: C.navy, fontFace: "Calibri",
    align: "left", margin: 0
  });
}
function divider(s, y = 0.9, color = C.violet) {
  s.addShape("rect", { x: 0.5, y, w: 9, h: 0.04, fill: { color }, line: { color } });
}
function chip(s, text, x, y, w = 1.8, bg = C.indigo) {
  s.addShape("roundRect", { x, y, w, h: 0.35, fill: { color: bg }, rectRadius: 0.05 });
  s.addText(text, { x, y: y + 0.04, w, h: 0.3, fontSize: 9, color: C.white,
    align: "center", bold: true, fontFace: "Calibri", margin: 0 });
}
function card(s, x, y, w, h, bg = C.card) {
  s.addShape("roundRect", { x, y, w, h,
    fill: { color: bg },
    shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.25 },
    rectRadius: 0.08 });
}
function pill(s, text, x, y, color = C.violet) {
  s.addShape("roundRect", { x, y, w: 0.18, h: 0.35, fill: { color }, rectRadius: 0.04 });
  s.addText(text, { x: x + 0.22, y: y + 0.04, w: 3.5, h: 0.28,
    fontSize: 12, color: C.white, bold: false, fontFace: "Calibri", margin: 0 });
}

// ═══════════════════════════════════════════════════════════════════════════════
async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title = "AcadAssist – AI-Powered Academic Assistant";

  // Pre-render icons
  const iBrain   = await ico(FaBrain,   "#A855F7");
  const iSearch  = await ico(FaSearch,  "#0891B2");
  const iDB      = await ico(FaDatabase,"#10B981");
  const iRobot   = await ico(FaRobot,   "#FFFFFF");
  const iChart   = await ico(FaChartBar,"#A855F7");
  const iCheck   = await ico(FaCheckCircle, "#10B981");
  const iWarn    = await ico(FaExclamationTriangle, "#F59E0B");
  const iLayer   = await ico(FaLayerGroup, "#0891B2");
  const iFile    = await ico(FaFileAlt, "#A855F7");
  const iCogs    = await ico(FaCogs,    "#94A3B8");
  const iLight   = await ico(FaLightbulb,"#F59E0B");
  const iCode    = await ico(FaCode,    "#A855F7");
  const iChip    = await ico(FaMicrochip,"#0891B2");
  const iGrad    = await ico(FaGraduationCap,"#FFFFFF");
  const iUniv    = await ico(FaUniversity,"#A855F7");
  const iFlask   = await ico(FaFlask,   "#10B981");
  const iShield  = await ico(FaShieldAlt,"#FFFFFF");
  const iSync    = await ico(FaSyncAlt, "#0891B2");
  const iFilter  = await ico(FaFilter,  "#A855F7");
  const iList    = await ico(FaListAlt, "#10B981");
  const iCloud   = await ico(FaCloud,   "#0891B2");
  const iArrow   = await ico(FaArrowRight,"#A855F7");

  // ─── Slide 1: TITLE ─────────────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    // Left deep bg panel
    s.addShape("rect", { x: 0, y: 0, w: 5.8, h: 5.625, fill: { color: C.dark } });
    // Accent bar
    s.addShape("rect", { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.violet } });
    // Right panel subtle
    s.addShape("rect", { x: 5.8, y: 0, w: 4.2, h: 5.625, fill: { color: C.navy } });

    // Decorative circles
    s.addShape("ellipse", { x: 6.2, y: -0.8, w: 3.5, h: 3.5,
      fill: { color: C.violet, transparency: 85 }, line: { color: C.violet, transparency: 70, width: 2 } });
    s.addShape("ellipse", { x: 7.0, y: 2.5, w: 2, h: 2,
      fill: { color: C.teal, transparency: 88 }, line: { color: C.teal, transparency: 70, width: 1 } });

    // Brain icon large
    s.addImage({ data: iBrain, x: 0.5, y: 0.65, w: 0.7, h: 0.7 });

    // Tag
    s.addText("FINAL YEAR PROJECT  ·  AI & ML", {
      x: 1.3, y: 0.72, w: 4.2, h: 0.35,
      fontSize: 9, color: C.vivid, bold: true, charSpacing: 3,
      fontFace: "Calibri", margin: 0
    });

    // Main title
    s.addText("AcadAssist", {
      x: 0.22, y: 1.3, w: 5.4, h: 1.2,
      fontSize: 56, bold: true, color: C.white,
      fontFace: "Calibri", margin: 0
    });
    s.addText("AI-Powered Academic Assistant", {
      x: 0.22, y: 2.45, w: 5.4, h: 0.55,
      fontSize: 20, color: C.vivid, fontFace: "Calibri", margin: 0
    });
    s.addText("for Sardar Patel Institute of Technology, Mumbai", {
      x: 0.22, y: 2.95, w: 5.4, h: 0.4,
      fontSize: 13, color: C.silver, fontFace: "Calibri", margin: 0
    });

    // Divider line
    s.addShape("rect", { x: 0.22, y: 3.45, w: 3.5, h: 0.035, fill: { color: C.violet } });

    // Team info
    s.addText("TEAM MEMBERS", { x: 0.22, y: 3.6, w: 3.8, h: 0.3,
      fontSize: 9, color: C.muted, charSpacing: 2, bold: true, fontFace: "Calibri", margin: 0 });
    s.addText([
      { text: "Presented by:", options: { color: C.silver, fontSize: 12 } },
      { text: "  B.Tech Final Year – Computer Science & Engineering", options: { color: C.white, fontSize: 12 } },
    ], { x: 0.22, y: 3.95, w: 5.4, h: 0.35, fontFace: "Calibri", margin: 0 });
    s.addText("Guide: Faculty Guide, Dept. of CSE  ·  AY 2025–26", {
      x: 0.22, y: 4.35, w: 5.4, h: 0.35,
      fontSize: 11, color: C.muted, fontFace: "Calibri", margin: 0
    });

    // Right side illustration elements
    s.addImage({ data: iGrad, x: 7.2, y: 1.0, w: 1.8, h: 1.8 });
    // Chip tags on right
    const tags = ["RAG Pipeline","Hybrid Retrieval","Groq LLM","FAISS","BM25","Reranking"];
    tags.forEach((t, i) => {
      const col = i % 2 === 0 ? C.indigo : C.cardL;
      chip(s, t, 6.0 + (i % 2) * 1.95, 3.1 + Math.floor(i/2) * 0.42, 1.85, col);
    });

    s.addNotes("Title slide. Introduce project name AcadAssist, institute SPIT Mumbai. Mention it is a RAG-based AI assistant for academic queries.");
  }

  // ─── Slide 2: AGENDA ────────────────────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "Presentation Roadmap");
    divider(s, 0.9, C.violet);

    const items = [
      ["01", "Problem Statement", "Why students struggle with academic info"],
      ["02", "Proposed Solution", "AcadAssist — the intelligent assistant"],
      ["03", "System Architecture", "End-to-end RAG pipeline design"],
      ["04", "Technology Stack", "Tools and frameworks powering RAKA"],
      ["05", "RAG Pipeline Deep Dive", "Chunking, embeddings, hybrid retrieval"],
      ["06", "Query Processing", "Normalization, routing, reranking"],
      ["07", "Implementation Status", "Done, in-progress, future scope"],
      ["08", "Challenges & Results", "Real engineering challenges solved"],
    ];
    items.forEach(([num, title, sub], i) => {
      const col = i % 2 === 0 ? 0.5 : 5.15;
      const row = Math.floor(i / 2);
      const y = 1.05 + row * 1.0;
      card(s, col, y, 4.5, 0.85, "FFFFFF");
      s.addShape("rect", { x: col, y, w: 0.06, h: 0.85, fill: { color: C.violet } });
      s.addText(num, { x: col + 0.12, y: y + 0.12, w: 0.5, h: 0.35,
        fontSize: 14, color: C.violet, bold: true, fontFace: "Calibri", margin: 0 });
      s.addText(title, { x: col + 0.65, y: y + 0.08, w: 3.7, h: 0.35,
        fontSize: 14, color: C.navy, bold: true, fontFace: "Calibri", margin: 0 });
      s.addText(sub, { x: col + 0.65, y: y + 0.48, w: 3.7, h: 0.28,
        fontSize: 10, color: C.muted, fontFace: "Calibri", margin: 0 });
    });
  }

  // ─── Slide 3: PROBLEM STATEMENT ─────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "The Problem Students Face");
    divider(s, 0.88, C.violet);

    // Big stat callout
    s.addShape("roundRect", { x: 0.5, y: 1.05, w: 2.3, h: 2.0,
      fill: { color: C.card }, rectRadius: 0.1,
      shadow: { type: "outer", blur: 10, offset: 3, angle: 135, color: "000000", opacity: 0.3 }});
    s.addText("3+", { x: 0.5, y: 1.2, w: 2.3, h: 0.9,
      fontSize: 52, bold: true, color: C.vivid, align: "center", fontFace: "Calibri", margin: 0 });
    s.addText("hours wasted\nsearching PDFs", { x: 0.5, y: 2.05, w: 2.3, h: 0.6,
      fontSize: 11, color: C.silver, align: "center", fontFace: "Calibri", margin: 0 });

    // Pain points
    const pains = [
      [iFile, "Scattered Documents", "Syllabus, regulations, notices — spread across dozens of PDFs with no unified search."],
      [iSearch, "No Semantic Search", "Keyword portals fail on natural queries. Students must know exact terms."],
      [iWarn, "Policy Misinterpretation", "Wrong attendance % or exam rules — sourced from seniors, not official docs."],
      [iBrain, "Information Overload", "Hundreds of pages of regulations. Students can't locate what matters quickly."],
    ];
    pains.forEach(([icon, title, body], i) => {
      const x = i < 2 ? 2.95 : 6.55;
      const y = i % 2 === 0 ? 1.05 : 2.85;
      card(s, x, y, 3.4, 1.65, C.card);
      s.addImage({ data: icon, x: x + 0.2, y: y + 0.22, w: 0.45, h: 0.45 });
      s.addText(title, { x: x + 0.75, y: y + 0.2, w: 2.55, h: 0.38,
        fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
      s.addText(body, { x: x + 0.2, y: y + 0.72, w: 3.0, h: 0.72,
        fontSize: 10.5, color: C.silver, fontFace: "Calibri", margin: 0 });
    });

    s.addText("Traditional portals & keyword search fail in academic environments — students deserve better.", {
      x: 0.5, y: 4.65, w: 9, h: 0.35,
      fontSize: 11, color: C.muted, italic: true, fontFace: "Calibri", margin: 0
    });
    s.addNotes("Problem: Students struggle with scattered PDFs, no semantic search, reliance on informal channels for official policy. 3+ hours lost searching. This leads to misinterpretation and academic risk.");
  }

  // ─── Slide 4: EXISTING LIMITATIONS ─────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "Limitations of Existing Systems");
    divider(s, 0.9, C.violet);

    // VS layout: old vs new
    // Old column
    s.addShape("roundRect", { x: 0.5, y: 1.05, w: 3.9, h: 4.15,
      fill: { color: "FEF2F2" }, rectRadius: 0.08 });
    s.addShape("rect", { x: 0.5, y: 1.05, w: 3.9, h: 0.55,
      fill: { color: C.red } });
    s.addText("❌  Without AcadAssist", { x: 0.55, y: 1.1, w: 3.8, h: 0.45,
      fontSize: 14, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });

    const oldItems = [
      "Keyword-only search — no semantic understanding",
      "Manual PDF reading — tedious and error-prone",
      "No conversational querying capability",
      "Scattered portals — no single source of truth",
      "LLM hallucinations on institution-specific facts",
      "No source citations — unverifiable answers",
    ];
    oldItems.forEach((item, i) => {
      s.addShape("ellipse", { x: 0.62, y: 1.75 + i * 0.54, w: 0.2, h: 0.2, fill: { color: C.red } });
      s.addText(item, { x: 0.9, y: 1.7 + i * 0.54, w: 3.3, h: 0.45,
        fontSize: 11, color: "7F1D1D", fontFace: "Calibri", margin: 0 });
    });

    // New column
    s.addShape("roundRect", { x: 5.65, y: 1.05, w: 3.9, h: 4.15,
      fill: { color: "F0FDF4" }, rectRadius: 0.08 });
    s.addShape("rect", { x: 5.65, y: 1.05, w: 3.9, h: 0.55,
      fill: { color: C.green } });
    s.addText("✓  With AcadAssist", { x: 5.7, y: 1.1, w: 3.8, h: 0.45,
      fontSize: 14, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });

    const newItems = [
      "Hybrid semantic + keyword retrieval (BM25 + SBERT)",
      "Conversational querying in natural language",
      "Grounded answers from official SPIT documents",
      "Single unified AI assistant for all queries",
      "Source-cited responses — every claim verifiable",
      "Query routing to right document category",
    ];
    newItems.forEach((item, i) => {
      s.addShape("ellipse", { x: 5.77, y: 1.75 + i * 0.54, w: 0.2, h: 0.2, fill: { color: C.green } });
      s.addText(item, { x: 6.05, y: 1.7 + i * 0.54, w: 3.35, h: 0.45,
        fontSize: 11, color: "14532D", fontFace: "Calibri", margin: 0 });
    });

    // VS badge
    s.addShape("ellipse", { x: 4.65, y: 2.55, w: 0.7, h: 0.7,
      fill: { color: C.violet },
      shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.2 }});
    s.addText("VS", { x: 4.65, y: 2.62, w: 0.7, h: 0.55,
      fontSize: 15, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
  }

  // ─── Slide 5: SOLUTION ──────────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "Introducing AcadAssist");
    divider(s, 0.88, C.violet);

    s.addText("An AI-powered academic knowledge assistant that answers student queries using only official SPIT documents — with full source citations.", {
      x: 0.5, y: 1.0, w: 9, h: 0.55,
      fontSize: 13.5, color: C.silver, fontFace: "Calibri", margin: 0, italic: true
    });

    const features = [
      [iBrain, "Semantic Search", "SBERT embeddings understand intent, not just keywords"],
      [iSearch, "Hybrid Retrieval", "BM25 + Dense fusion for maximum recall on all query types"],
      [iDB, "Grounded Answers", "Responses drawn exclusively from official SPIT documents"],
      [iRobot, "LLM Generation", "Groq Llama 3.3-70B generates concise, accurate answers"],
      [iFilter, "Smart Routing", "Queries auto-classified: exam / syllabus / placement / activity"],
      [iCheck, "Source Citations", "Every answer cites document name and page numbers"],
    ];
    features.forEach(([icon, title, body], i) => {
      const x = 0.5 + (i % 3) * 3.1;
      const y = 1.7 + Math.floor(i / 3) * 1.7;
      card(s, x, y, 2.9, 1.52, C.card);
      s.addImage({ data: icon, x: x + 0.2, y: y + 0.18, w: 0.45, h: 0.45 });
      s.addText(title, { x: x + 0.75, y: y + 0.18, w: 2.0, h: 0.38,
        fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
      s.addText(body, { x: x + 0.2, y: y + 0.72, w: 2.6, h: 0.62,
        fontSize: 10, color: C.silver, fontFace: "Calibri", margin: 0 });
    });
    s.addNotes("AcadAssist = RAG pipeline tuned to SPIT institutional documents. Hybrid retrieval, source-grounded, Groq LLM, full citation. Not a general chatbot — domain-specific intelligence.");
  }

  // ─── Slide 6: OBJECTIVES ────────────────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "Project Objectives");
    divider(s, 0.9, C.violet);

    const objs = [
      ["01", FaGraduationCap, "Improve Accessibility", "Make institutional information instantly accessible to all SPIT students via natural language queries."],
      ["02", FaSearch,        "Semantic Retrieval",    "Enable intent-aware search over examination rules, syllabus, and placement policies."],
      ["03", FaShieldAlt,     "Grounded Responses",   "Guarantee every answer is sourced from official documents — eliminating hallucinations."],
      ["04", FaLayerGroup,    "Hybrid Retrieval",      "Combine BM25 keyword and SBERT dense retrieval via RRF for superior recall."],
      ["05", FaFilter,        "Query Intelligence",    "Auto-classify and route queries to the most relevant document category."],
      ["06", FaChartBar,      "Scalable Architecture", "Build a modular, extensible RAG system ready for multi-department deployment."],
    ];
    objs.forEach(async ([num, , title, body], i) => {
      const x = i % 2 === 0 ? 0.5 : 5.2;
      const y = 1.05 + Math.floor(i / 2) * 1.35;
      const icD = await ico(objs[i][1], "#7B2FBE");
      card(s, x, y, 4.55, 1.18, "FFFFFF");
      s.addShape("rect", { x, y, w: 0.06, h: 1.18, fill: { color: C.violet } });
      s.addImage({ data: icD, x: x + 0.18, y: y + 0.15, w: 0.45, h: 0.45 });
      s.addText(`${num}  ${title}`, { x: x + 0.72, y: y + 0.12, w: 3.7, h: 0.4,
        fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", margin: 0 });
      s.addText(body, { x: x + 0.72, y: y + 0.55, w: 3.72, h: 0.52,
        fontSize: 10.5, color: C.muted, fontFace: "Calibri", margin: 0 });
    });
    s.addNotes("Six core objectives: accessibility, semantic retrieval, grounded responses, hybrid retrieval, query intelligence, scalability.");
  }

  // ─── Slide 7: ARCHITECTURE ──────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "System Architecture");
    divider(s, 0.88, C.violet);

    // Flow pipeline boxes
    const flow = [
      { label: "User Query", sub: "Natural Language", color: C.teal,    x: 0.28, y: 1.05 },
      { label: "Normalization", sub: "Abbrev. Expand",color: C.indigo,  x: 0.28, y: 1.82 },
      { label: "Classification", sub: "Category Route",color: C.violet, x: 0.28, y: 2.59 },
      { label: "BM25",          sub: "Keyword",        color: C.teal,   x: 0.28, y: 3.36 },
      { label: "FAISS Dense",   sub: "Semantic",       color: C.violet, x: 0.28, y: 4.13 },
    ];
    flow.forEach(({label, sub, color, x, y}) => {
      s.addShape("roundRect", { x, y, w: 2.05, h: 0.65,
        fill: { color },
        shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.25 },
        rectRadius: 0.06 });
      s.addText(label, { x, y: y + 0.04, w: 2.05, h: 0.3,
        fontSize: 11, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(sub, { x, y: y + 0.35, w: 2.05, h: 0.22,
        fontSize: 8, color: "E0E7FF", align: "center", fontFace: "Calibri", margin: 0 });
    });

    // Arrows between left boxes
    [1.7, 2.47, 3.24].forEach(y => {
      s.addShape("rect", { x: 1.2, y, w: 0.03, h: 0.12, fill: { color: C.muted } });
      s.addText("▾", { x: 1.12, y: y + 0.07, w: 0.2, h: 0.2, fontSize: 10, color: C.muted,
        align: "center", fontFace: "Calibri", margin: 0 });
    });

    // Merge box
    s.addShape("roundRect", { x: 2.55, y: 3.45, w: 1.65, h: 1.2,
      fill: { color: "1A3060" }, rectRadius: 0.06 });
    s.addText("RRF\nFusion", { x: 2.55, y: 3.65, w: 1.65, h: 0.65,
      fontSize: 13, bold: true, color: C.vivid, align: "center", fontFace: "Calibri", margin: 0 });
    s.addText("Rank Fusion", { x: 2.55, y: 4.25, w: 1.65, h: 0.25,
      fontSize: 8, color: C.muted, align: "center", fontFace: "Calibri", margin: 0 });
    // Lines to RRF
    s.addShape("line", { x: 2.33, y: 3.69, w: 0.22, h: 0, line: { color: C.muted, width: 1.5 } });
    s.addShape("line", { x: 2.33, y: 4.46, w: 0.22, h: 0, line: { color: C.muted, width: 1.5 } });

    // Right pipeline
    const rflow = [
      { label: "Reranker",    sub: "CrossEncoder", color: C.violet, x: 4.45, y: 1.05 },
      { label: "Context",     sub: "Builder",      color: C.indigo, x: 4.45, y: 1.85 },
      { label: "Groq LLM",   sub: "Llama 3.3-70B",color: "7C3AED", x: 4.45, y: 2.65 },
      { label: "Response",   sub: "+ Sources",     color: C.green,  x: 4.45, y: 3.45 },
    ];
    rflow.forEach(({label, sub, color, x, y}) => {
      s.addShape("roundRect", { x, y, w: 2.0, h: 0.65,
        fill: { color },
        shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.25 },
        rectRadius: 0.06 });
      s.addText(label, { x, y: y + 0.04, w: 2.0, h: 0.3,
        fontSize: 11, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(sub, { x, y: y + 0.35, w: 2.0, h: 0.22,
        fontSize: 8, color: "E0E7FF", align: "center", fontFace: "Calibri", margin: 0 });
    });
    [1.70, 2.50, 3.30].forEach(y => {
      s.addText("▾", { x: 5.35, y, w: 0.2, h: 0.2, fontSize: 10, color: C.muted,
        align: "center", fontFace: "Calibri", margin: 0 });
    });

    // Arrow from RRF to Reranker
    s.addShape("line", { x: 4.2, y: 3.95, w: 0.25, h: 0, line: { color: C.muted, width: 1.5 } });

    // Ingestion pipeline (right side)
    s.addShape("roundRect", { x: 7.0, y: 1.05, w: 2.75, h: 3.85,
      fill: { color: C.card }, rectRadius: 0.1 });
    s.addText("DOCUMENT INGESTION", { x: 7.1, y: 1.12, w: 2.55, h: 0.3,
      fontSize: 9, bold: true, color: C.vivid, charSpacing: 1, align: "center", fontFace: "Calibri", margin: 0 });

    const ingestion = ["PDF Loader", "Chunker\n300-500 words", "SBERT\nEmbedder", "FAISS\nVector Store"];
    ingestion.forEach((label, i) => {
      const iy = 1.55 + i * 0.75;
      s.addShape("roundRect", { x: 7.15, y: iy, w: 2.45, h: 0.58,
        fill: { color: C.cardL }, rectRadius: 0.05 });
      s.addText(label, { x: 7.15, y: iy + 0.08, w: 2.45, h: 0.42,
        fontSize: 10.5, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
      if (i < 3) s.addText("▾", { x: 8.3, y: iy + 0.58, w: 0.2, h: 0.2,
        fontSize: 9, color: C.muted, align: "center", fontFace: "Calibri", margin: 0 });
    });

    // Bottom label
    s.addText("Offline Phase", { x: 7.0, y: 4.98, w: 2.75, h: 0.3,
      fontSize: 9, color: C.muted, italic: true, align: "center", fontFace: "Calibri", margin: 0 });

    s.addNotes("End-to-end pipeline: query normalization → classification → hybrid retrieval (BM25+FAISS via RRF) → cross-encoder reranking → context building → Groq LLM → grounded response with sources. Offline ingestion: PDF → chunk → embed → FAISS index.");
  }

  // ─── Slide 8: TECH STACK ────────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "Technology Stack");
    divider(s, 0.88, C.violet);

    const groups = [
      { cat: "AI & ML Core", color: C.violet, items: [
        ["Sentence Transformers", "all-MiniLM-L6-v2 · 384-dim embeddings"],
        ["CrossEncoder", "ms-marco-MiniLM reranking"],
        ["Groq API", "Llama 3.3-70B-Versatile LLM"],
      ]},
      { cat: "Retrieval & Storage", color: C.teal, items: [
        ["FAISS / NumpyIP", "Exact inner-product vector search"],
        ["BM25", "Pure-Python Okapi BM25 keyword retrieval"],
        ["Pickle / NumPy", "Metadata & embedding persistence"],
      ]},
      { cat: "Document Processing", color: C.indigo, items: [
        ["pdfplumber", "PDF text & table extraction"],
        ["PyPDF", "Fallback PDF reading"],
        ["Custom Chunker", "Sentence-boundary aware 300-500w chunks"],
      ]},
      { cat: "Frontend & Infra", color: C.green, items: [
        ["Streamlit", "Interactive Python-native web UI"],
        ["Python 3.10+", "Core language & pipeline orchestration"],
        ["dotenv / urllib", "Config management & HTTP API calls"],
      ]},
    ];

    groups.forEach(({cat, color, items}, gi) => {
      const x = gi % 2 === 0 ? 0.4 : 5.2;
      const y = gi < 2 ? 1.05 : 3.2;
      s.addShape("roundRect", { x, y, w: 4.6, h: 1.95,
        fill: { color: C.card }, rectRadius: 0.08,
        shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.25 }});
      s.addShape("rect", { x, y, w: 4.6, h: 0.42, fill: { color } });
      s.addText(cat, { x: x + 0.12, y: y + 0.08, w: 4.3, h: 0.28,
        fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
      items.forEach(([name, desc], ii) => {
        const iy = y + 0.55 + ii * 0.44;
        s.addShape("ellipse", { x: x + 0.15, y: iy + 0.1, w: 0.14, h: 0.14, fill: { color } });
        s.addText(name, { x: x + 0.38, y: iy, w: 1.5, h: 0.35,
          fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
        s.addText(desc, { x: x + 1.95, y: iy, w: 2.5, h: 0.35,
          fontSize: 10, color: C.muted, fontFace: "Calibri", margin: 0 });
      });
    });
    s.addNotes("Four categories: AI & ML Core (SBERT, CrossEncoder, Groq), Retrieval (FAISS, BM25), Document Processing (pdfplumber, custom chunker), Frontend (Streamlit).");
  }

  // ─── Slide 9: RAG PIPELINE DEEP DIVE ────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "RAG Pipeline — Deep Dive");
    divider(s, 0.9, C.violet);

    const steps = [
      { n:"1", title:"PDF Ingestion", body:"pdfplumber extracts text & tables. Boilerplate stripped. Doc type auto-detected.", color: C.indigo },
      { n:"2", title:"Smart Chunking", body:"300–500 word sentence-boundary chunks with section metadata, page numbers & doc_type.", color: C.violet },
      { n:"3", title:"SBERT Embedding", body:"all-MiniLM-L6-v2 produces 384-dim L2-normalised vectors. Stored in FAISS NumpyIP index.", color: C.teal },
      { n:"4", title:"Hybrid Retrieval", body:"BM25 (k1=1.5, b=0.75) keyword + FAISS dense retrieval fused via Reciprocal Rank Fusion.", color: "7C3AED" },
      { n:"5", title:"CrossEncoder Rerank", body:"ms-marco-MiniLM scores (query, chunk) pairs jointly. Top-5 sent to LLM.", color: C.green },
      { n:"6", title:"LLM Generation", body:"Groq Llama 3.3-70B generates answer from structured context. MAX_CONTEXT=12,000 chars.", color: C.amber },
    ];
    steps.forEach(({n, title, body, color}, i) => {
      const x = 0.4 + (i % 3) * 3.18;
      const y = i < 3 ? 1.05 : 3.15;
      card(s, x, y, 3.0, 1.85, "FFFFFF");
      s.addShape("rect", { x, y, w: 3.0, h: 0.06, fill: { color } });
      s.addShape("ellipse", { x: x + 0.18, y: y + 0.22, w: 0.42, h: 0.42, fill: { color } });
      s.addText(n, { x: x + 0.18, y: y + 0.24, w: 0.42, h: 0.38,
        fontSize: 16, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(title, { x: x + 0.72, y: y + 0.22, w: 2.15, h: 0.38,
        fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", margin: 0 });
      s.addText(body, { x: x + 0.18, y: y + 0.75, w: 2.72, h: 0.9,
        fontSize: 10.5, color: C.muted, fontFace: "Calibri", margin: 0 });
      // Arrow
      if (i % 3 < 2) {
        s.addText("›", { x: x + 3.02, y: y + 0.7, w: 0.18, h: 0.45,
          fontSize: 22, color: color, align: "center", fontFace: "Calibri", margin: 0 });
      }
    });
    s.addText("Why Hybrid? BM25 catches exact-term queries (FF, ATKT, CGPA). SBERT handles semantic queries. RRF combines both ranked lists for best-of-both-worlds retrieval.", {
      x: 0.4, y: 5.05, w: 9.2, h: 0.38,
      fontSize: 11, color: C.violet, italic: true, bold: true, fontFace: "Calibri", margin: 0
    });
    s.addNotes("6-stage pipeline. Key insight: BM25 alone fails on semantic queries; SBERT alone misses exact acronyms like FF, ATKT. RRF fusion solves both. CrossEncoder reranking boosts precision.");
  }

  // ─── Slide 10: QUERY PROCESSING ─────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "Query Processing Workflow");
    divider(s, 0.88, C.violet);

    // Example query at top
    s.addShape("roundRect", { x: 0.5, y: 1.05, w: 9, h: 0.65,
      fill: { color: C.card }, rectRadius: 0.08 });
    s.addText("Example Query:", { x: 0.75, y: 1.12, w: 1.5, h: 0.3,
      fontSize: 10, color: C.muted, fontFace: "Calibri", margin: 0 });
    s.addText('"what is cg criteria for elite spit placement?"', { x: 2.3, y: 1.12, w: 7.0, h: 0.42,
      fontSize: 14, bold: true, color: C.vivid, fontFace: "Calibri", margin: 0 });

    // Steps in a horizontal pipeline
    const steps = [
      { step:"Step 1", title:"Normalize", detail:"'cg' → 'cgpa'\n'spit' → 'sardar patel institute of technology'", color: C.teal },
      { step:"Step 2", title:"Classify", detail:"Keyword match:\n'placement', 'elite' → category: placement", color: C.violet },
      { step:"Step 3", title:"Retrieve", detail:"BM25 top-20 + FAISS top-20\nRRF fusion → ranked candidates", color: C.indigo },
      { step:"Step 4", title:"Rerank", detail:"CrossEncoder scores\n(query, chunk) pairs → top-5", color: "7C3AED" },
      { step:"Step 5", title:"Generate", detail:"Groq LLM + context\nblock (12,000 chars max)", color: C.green },
    ];
    steps.forEach(({step, title, detail, color}, i) => {
      const x = 0.4 + i * 1.84;
      card(s, x, 1.9, 1.75, 2.55, C.card);
      s.addShape("roundRect", { x: x + 0.55, y: 1.95, w: 0.65, h: 0.62,
        fill: { color }, rectRadius: 0.06 });
      s.addText((i+1).toString(), { x: x + 0.55, y: 2.0, w: 0.65, h: 0.52,
        fontSize: 18, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(step, { x, y: 2.62, w: 1.75, h: 0.28,
        fontSize: 9, color: color, bold: true, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(title, { x, y: 2.88, w: 1.75, h: 0.32,
        fontSize: 12, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(detail, { x: x + 0.1, y: 3.3, w: 1.55, h: 0.95,
        fontSize: 9, color: C.silver, align: "center", fontFace: "Calibri", margin: 0 });
      if (i < 4) s.addText("›", { x: x + 1.75, y: 2.5, w: 0.18, h: 0.45,
        fontSize: 22, color: C.muted, align: "center", fontFace: "Calibri", margin: 0 });
    });

    // Output box
    s.addShape("roundRect", { x: 0.5, y: 4.58, w: 9, h: 0.72,
      fill: { color: C.card }, rectRadius: 0.08 });
    s.addText("✓  Output:", { x: 0.75, y: 4.65, w: 1.0, h: 0.32,
      fontSize: 11, color: C.green, bold: true, fontFace: "Calibri", margin: 0 });
    s.addText('"Elite category requires CGPA ≥ 7.5 (T&P Policy 2025-26, pg. 4, score=0.874)"', {
      x: 1.75, y: 4.65, w: 7.5, h: 0.42,
      fontSize: 12, color: C.vivid, fontFace: "Calibri", margin: 0 });

    s.addNotes("Walk through a real query. Show abbreviation expansion, category routing to placement docs, retrieval, reranking, and final cited answer.");
  }

  // ─── Slide 11: UI / FRONTEND ─────────────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "Frontend — User Interface");
    divider(s, 0.9, C.violet);

    // Left: UI mockup
    // Chat window
    s.addShape("roundRect", { x: 0.4, y: 1.05, w: 5.2, h: 4.25,
      fill: { color: C.navy }, rectRadius: 0.12,
      shadow: { type: "outer", blur: 12, offset: 3, angle: 135, color: "000000", opacity: 0.3 }});
    s.addShape("rect", { x: 0.4, y: 1.05, w: 5.2, h: 0.5, fill: { color: C.dark } });
    s.addText("🤖  AcadAssist  ·  SPIT Academic AI", { x: 0.55, y: 1.1, w: 4.8, h: 0.38,
      fontSize: 11, color: C.white, bold: true, fontFace: "Calibri", margin: 0 });

    // User bubble
    s.addShape("roundRect", { x: 2.5, y: 1.72, w: 2.9, h: 0.52,
      fill: { color: C.violet }, rectRadius: 0.08 });
    s.addText("What are sem 5 subjects for CSE?", { x: 2.55, y: 1.8, w: 2.8, h: 0.35,
      fontSize: 9.5, color: C.white, fontFace: "Calibri", margin: 0 });

    // AI response bubble
    s.addShape("roundRect", { x: 0.55, y: 2.38, w: 4.7, h: 1.28,
      fill: { color: C.card }, rectRadius: 0.08 });
    s.addText("Semester 5 CSE courses include: Data Structures & Algorithms (DSA - 4 credits), Database Management Systems (DBMS - 4 credits), Theory of Computation (TOC - 3 credits)...\n[Source: cse-syllabus.pdf, pp. 34-36, score: 0.921]", {
      x: 0.65, y: 2.45, w: 4.5, h: 1.1,
      fontSize: 9, color: C.silver, fontFace: "Calibri", margin: 0 });

    // Input box
    s.addShape("roundRect", { x: 0.55, y: 4.65, w: 4.7, h: 0.45,
      fill: { color: C.cardL }, rectRadius: 0.06 });
    s.addText("Ask anything about SPIT academics...", { x: 0.7, y: 4.72, w: 3.8, h: 0.3,
      fontSize: 9.5, color: C.muted, fontFace: "Calibri", margin: 0 });
    s.addShape("roundRect", { x: 5.0, y: 4.65, w: 0.5, h: 0.45,
      fill: { color: C.violet }, rectRadius: 0.06 });
    s.addText("→", { x: 5.0, y: 4.69, w: 0.5, h: 0.38,
      fontSize: 13, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });

    // Right: feature list
    const feats = [
      [iChat = iRobot, "Conversational Chat", "Natural back-and-forth querying"],
      [iFile, "Source Cards", "Expandable document references with page numbers"],
      [iFilter, "Quick Categories", "One-click: Exams · Syllabus · Placement · Activity"],
      [iCheck, "Cited Answers", "Every response shows document + page + score"],
    ];
    const iChatD = await ico(FaRobot, "#7B2FBE");
    const iFileD = await ico(FaFileAlt, "#0891B2");
    const iFilterD = await ico(FaFilter, "#10B981");
    const iCheckD = await ico(FaCheckCircle, "#F59E0B");
    const featIcons = [iChatD, iFileD, iFilterD, iCheckD];
    const featTitles = ["Conversational Chat", "Source Cards", "Quick Categories", "Cited Answers"];
    const featSubs = [
      "Natural back-and-forth querying",
      "Expandable references with page numbers",
      "One-click: Exams · Syllabus · Placement",
      "Document + page + similarity score shown",
    ];
    featIcons.forEach((ic, i) => {
      const y = 1.12 + i * 1.04;
      card(s, 5.85, y, 3.8, 0.88, "FFFFFF");
      s.addImage({ data: ic, x: 6.0, y: y + 0.18, w: 0.42, h: 0.42 });
      s.addText(featTitles[i], { x: 6.55, y: y + 0.12, w: 2.95, h: 0.35,
        fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", margin: 0 });
      s.addText(featSubs[i], { x: 6.55, y: y + 0.5, w: 2.95, h: 0.28,
        fontSize: 10, color: C.muted, fontFace: "Calibri", margin: 0 });
    });
    s.addText("Built with Streamlit · Modern Python-native frontend", {
      x: 5.85, y: 5.15, w: 3.8, h: 0.28,
      fontSize: 9.5, color: C.muted, italic: true, fontFace: "Calibri", margin: 0
    });
    s.addNotes("Streamlit-based chat interface. Shows user bubbles, AI response with citations, source cards. Feature list: conversational, source cards, quick categories, cited answers.");
  }

  // ─── Slide 12: IMPLEMENTATION STATUS ────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "Implementation Status");
    divider(s, 0.88, C.violet);

    // Three columns
    const cols = [
      { label: "✅  Completed", color: C.green, bg: "0A2A1A", x: 0.3, items: [
        "Hybrid BM25 + Dense Retrieval",
        "FAISS NumpyIP Vector Store",
        "SBERT Embedding (all-MiniLM-L6-v2)",
        "CrossEncoder Reranking",
        "Query Normalization & Classification",
        "Groq Llama 3.3-70B Integration",
        "Multi-PDF Document Ingestion",
        "Sentence-aware Chunking Pipeline",
        "Source Citation in Responses",
        "Streamlit Frontend (partial)",
      ]},
      { label: "🔧  In Progress", color: C.amber, bg: "2A1E0A", x: 3.55, items: [
        "Frontend UI Polish",
        "Response Quality Optimization",
        "Advanced Prompt Engineering",
        "Latency Profiling",
        "Integration Testing",
      ]},
      { label: "🚀  Future Scope", color: C.vivid, bg: "1A0A2A", x: 6.8, items: [
        "Faithfulness Verification (LR)",
        "Hallucination Detection Layer",
        "Production Deployment",
        "Role-based Authentication",
        "Multilingual Support",
        "Voice Interface",
        "Analytics Dashboard",
        "OCR for Scanned PDFs",
      ]},
    ];

    cols.forEach(({label, color, bg, x, items}) => {
      const h = 0.5 + items.length * 0.43;
      s.addShape("roundRect", { x, y: 1.05, w: 3.05, h: 4.35,
        fill: { color: bg }, rectRadius: 0.08 });
      s.addShape("roundRect", { x, y: 1.05, w: 3.05, h: 0.48,
        fill: { color }, rectRadius: 0.08 });
      s.addText(label, { x: x + 0.1, y: 1.12, w: 2.85, h: 0.35,
        fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
      items.forEach((item, i) => {
        s.addShape("ellipse", { x: x + 0.15, y: 1.68 + i * 0.43, w: 0.14, h: 0.14, fill: { color } });
        s.addText(item, { x: x + 0.38, y: 1.62 + i * 0.43, w: 2.55, h: 0.37,
          fontSize: 10.5, color: C.silver, fontFace: "Calibri", margin: 0 });
      });
    });
    s.addNotes("Core RAG pipeline is fully functional. Frontend is partially done. Faithfulness verification, deployment, and advanced features are future scope. Be clear to evaluators about what is done vs planned.");
  }

  // ─── Slide 13: CHALLENGES ───────────────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "Engineering Challenges & Solutions");
    divider(s, 0.9, C.violet);

    const challenges = [
      { prob: "Exact-Term Retrieval Failure", sol: "BM25 hybrid retrieval — acronyms like FF, ATKT, CGPA now reliably retrieved via keyword scoring" },
      { prob: "Short Query Semantic Mismatch", sol: "Query normalization expands 'cg', 'sem', 'spit' before embedding — bridging vocabulary gap" },
      { prob: "PDF Boilerplate Noise", sol: "Regex-based boilerplate stripping per doc type (exam vs syllabus) improves chunk signal quality" },
      { prob: "LLM Context Truncation", sol: "MAX_CONTEXT_CHARS doubled to 12,000; fair allocation across all retrieved chunks" },
      { prob: "Table Data Loss", sol: "pdfplumber table extraction with pipe-delimited row format preserves tabular content in chunks" },
      { prob: "No Reranking Precision", sol: "CrossEncoder reranker rescores (query, chunk) jointly — precision improved over bi-encoder alone" },
    ];
    challenges.forEach(({prob, sol}, i) => {
      const x = i % 2 === 0 ? 0.4 : 5.15;
      const y = 1.05 + Math.floor(i / 2) * 1.42;
      card(s, x, y, 4.6, 1.28, "FFFFFF");
      s.addShape("rect", { x, y, w: 0.06, h: 1.28, fill: { color: C.red } });
      s.addText("CHALLENGE", { x: x + 0.18, y: y + 0.08, w: 1.2, h: 0.2,
        fontSize: 7.5, color: C.red, bold: true, charSpacing: 1, fontFace: "Calibri", margin: 0 });
      s.addText(prob, { x: x + 0.18, y: y + 0.25, w: 4.25, h: 0.32,
        fontSize: 12, bold: true, color: C.navy, fontFace: "Calibri", margin: 0 });
      s.addShape("rect", { x: x + 0.18, y: y + 0.61, w: 4.22, h: 0.022, fill: { color: "E2E8F0" }});
      s.addText("✓  " + sol, { x: x + 0.18, y: y + 0.7, w: 4.25, h: 0.48,
        fontSize: 10.5, color: "166534", fontFace: "Calibri", margin: 0 });
    });
    s.addNotes("Six real engineering challenges encountered and solved. Shows depth of implementation: retrieval failures, PDF processing, context limits, table handling, precision improvements.");
  }

  // ─── Slide 14: RESULTS ──────────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "Results & Sample Queries");
    divider(s, 0.88, C.violet);

    const qs = [
      { q: "What is the minimum attendance to appear for exams?", src: "examination.pdf · pp. 8-9", ans: "Students must maintain ≥ 75% attendance in each subject to be eligible for end-semester examinations." },
      { q: "What courses are offered in Semester 5 for CSE?", src: "cse-syllabus.pdf · pp. 34-36", ans: "Sem 5 includes: Data Structures, DBMS, Theory of Computation, Operating Systems, and one open elective." },
      { q: "What is the CGPA requirement for Elite placement category?", src: "placement-policy.pdf · pp. 3-4", ans: "Elite category requires CGPA ≥ 7.5 to be eligible. Dream companies require CGPA ≥ 8.0." },
    ];
    qs.forEach(({q, src, ans}, i) => {
      const y = 1.05 + i * 1.45;
      card(s, 0.4, y, 9.2, 1.3, C.card);
      s.addImage({ data: iSearch, x: 0.55, y: y + 0.2, w: 0.38, h: 0.38 });
      s.addText(q, { x: 1.05, y: y + 0.14, w: 7.5, h: 0.38,
        fontSize: 12.5, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
      s.addText("📄 " + src, { x: 1.05, y: y + 0.52, w: 3.5, h: 0.26,
        fontSize: 9.5, color: C.teal, fontFace: "Calibri", margin: 0 });
      s.addText(ans, { x: 1.05, y: y + 0.78, w: 8.3, h: 0.42,
        fontSize: 10.5, color: C.silver, fontFace: "Calibri", margin: 0 });
    });
    s.addNotes("Three real example queries with grounded answers and source citations. Demonstrates working system across all document categories.");
  }

  // ─── Slide 15: FUTURE SCOPE ──────────────────────────────────────────────────
  {
    const s = lightSlide(pres);
    slideTitleDark(s, "Future Enhancements");
    divider(s, 0.9, C.violet);

    const phases = [
      { phase: "Phase 2  ·  Near-Term", color: C.violet, items: [
        { icon: iShield, title: "Faithfulness Verification", body: "Logistic Regression classifier on (answer, context) pairs — semantic + keyword + numeric features" },
        { icon: iChart, title: "Evaluation Dashboard", body: "ROUGE-L, Precision@K, hallucination rate, latency P95 tracked in real-time" },
      ]},
      { phase: "Phase 3  ·  Production", color: C.teal, items: [
        { icon: iCloud, title: "Cloud Deployment", body: "Docker + Nginx + Uvicorn on university server with JWT auth and department-level access control" },
        { icon: iList, title: "Multilingual Support", body: "paraphrase-multilingual-MiniLM for Hindi / Marathi academic queries" },
      ]},
      { phase: "Phase 4  ·  Advanced", color: C.indigo, items: [
        { icon: iChip, title: "Fine-tuned LLM", body: "Domain-specific Llama 3 fine-tuned on SPIT QA pairs — reduces API dependency" },
        { icon: iSync, title: "Conversational Memory", body: "Session-level history for follow-up questions and multi-turn academic guidance" },
      ]},
    ];

    phases.forEach(({phase, color, items}, pi) => {
      const y = 1.05 + pi * 1.45;
      s.addShape("roundRect", { x: 0.4, y, w: 9.2, h: 1.32,
        fill: { color: "FFFFFF" }, rectRadius: 0.08,
        shadow: { type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.1 }});
      s.addShape("rect", { x: 0.4, y, w: 0.06, h: 1.32, fill: { color } });
      s.addText(phase, { x: 0.55, y: y + 0.08, w: 2.0, h: 0.28,
        fontSize: 9.5, bold: true, color, charSpacing: 1, fontFace: "Calibri", margin: 0 });
      items.forEach(({icon, title, body}, ii) => {
        const x = ii === 0 ? 0.55 : 5.15;
        s.addImage({ data: icon, x, y: y + 0.42, w: 0.35, h: 0.35 });
        s.addText(title, { x: x + 0.45, y: y + 0.38, w: 4.0, h: 0.32,
          fontSize: 12, bold: true, color: C.navy, fontFace: "Calibri", margin: 0 });
        s.addText(body, { x: x + 0.45, y: y + 0.72, w: 4.3, h: 0.45,
          fontSize: 9.5, color: C.muted, fontFace: "Calibri", margin: 0 });
      });
    });
    s.addNotes("Three-phase roadmap. Phase 2: faithfulness verification and evaluation. Phase 3: production deployment and multilingual. Phase 4: fine-tuning and conversational memory.");
  }

  // ─── Slide 16: CONCLUSION ────────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.dark } });
    slideTitle(s, "Conclusion");
    divider(s, 0.88, C.violet);

    s.addText("AcadAssist demonstrates that a well-engineered RAG system can transform how students interact with institutional knowledge — replacing hours of PDF hunting with instant, cited answers.", {
      x: 0.5, y: 1.05, w: 9, h: 0.72,
      fontSize: 14, color: C.silver, fontFace: "Calibri", margin: 0
    });

    const wins = [
      { n: "6", label: "Core modules\nfully built" },
      { n: "4+", label: "Document types\nindexed" },
      { n: "100%", label: "Answers\nsource-grounded" },
      { n: "<3s", label: "Target end-to-end\nlatency" },
    ];
    wins.forEach(({n, label}, i) => {
      const x = 0.5 + i * 2.28;
      s.addShape("roundRect", { x, y: 1.95, w: 2.1, h: 1.52,
        fill: { color: C.card }, rectRadius: 0.1,
        shadow: { type: "outer", blur: 8, offset: 2, angle: 135, color: "000000", opacity: 0.25 }});
      s.addText(n, { x, y: 2.05, w: 2.1, h: 0.72,
        fontSize: 38, bold: true, color: C.vivid, align: "center", fontFace: "Calibri", margin: 0 });
      s.addText(label, { x, y: 2.75, w: 2.1, h: 0.5,
        fontSize: 11, color: C.silver, align: "center", fontFace: "Calibri", margin: 0 });
    });

    s.addText("Key Contributions", { x: 0.5, y: 3.65, w: 4, h: 0.38,
      fontSize: 14, bold: true, color: C.white, fontFace: "Calibri", margin: 0 });
    const contribs = [
      "Pure-Python BM25 + SBERT hybrid retrieval with RRF fusion",
      "Query normalization for academic abbreviation expansion",
      "Category-aware routing across 4 document domains",
      "Modular, extensible architecture for multi-department deployment",
    ];
    contribs.forEach((c, i) => {
      s.addShape("rect", { x: 0.5, y: 4.12 + i * 0.33, w: 0.18, h: 0.18, fill: { color: C.violet } });
      s.addText(c, { x: 0.78, y: 4.07 + i * 0.33, w: 8.6, h: 0.3,
        fontSize: 11.5, color: C.silver, fontFace: "Calibri", margin: 0 });
    });
    s.addNotes("Summarize: 6 modules built, 4+ document types, all answers source-grounded. Key contributions: hybrid retrieval, query normalization, category routing, modular design.");
  }

  // ─── Slide 17: THANK YOU ─────────────────────────────────────────────────────
  {
    const s = darkSlide(pres);
    s.addShape("rect", { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.dark } });
    // Decorative circles
    s.addShape("ellipse", { x: -1.0, y: -0.5, w: 4, h: 4,
      fill: { color: C.violet, transparency: 88 }, line: { color: C.violet, transparency: 75, width: 2 } });
    s.addShape("ellipse", { x: 7.5, y: 3.0, w: 3.5, h: 3.5,
      fill: { color: C.teal, transparency: 88 }, line: { color: C.teal, transparency: 75, width: 2 } });
    s.addShape("ellipse", { x: 8.0, y: 0.5, w: 2.0, h: 2.0,
      fill: { color: C.vivid, transparency: 92 } });

    s.addImage({ data: iBrain, x: 4.6, y: 0.5, w: 0.85, h: 0.85 });

    s.addText("AcadAssist", { x: 1.0, y: 1.5, w: 8, h: 1.1,
      fontSize: 54, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
    s.addText("AI-Powered Academic Assistant for SPIT", { x: 1.0, y: 2.55, w: 8, h: 0.48,
      fontSize: 18, color: C.vivid, align: "center", fontFace: "Calibri", margin: 0 });

    s.addShape("rect", { x: 3.5, y: 3.1, w: 3.0, h: 0.04, fill: { color: C.violet } });

    s.addText("Thank You", { x: 1.0, y: 3.28, w: 8, h: 0.72,
      fontSize: 38, bold: true, color: C.white, align: "center", fontFace: "Calibri", margin: 0 });
    s.addText("Questions & Discussion", { x: 1.0, y: 3.98, w: 8, h: 0.42,
      fontSize: 18, color: C.silver, align: "center", fontFace: "Calibri", margin: 0 });

    // Tags
    const tags2 = ["RAG", "SBERT", "BM25", "FAISS", "Groq", "Python"];
    tags2.forEach((t, i) => {
      chip(s, t, 2.2 + i * 1.0, 4.68, 0.92, i % 2 === 0 ? C.indigo : C.cardL);
    });

    s.addText("Sardar Patel Institute of Technology, Mumbai  ·  B.Tech CSE  ·  2025–26", {
      x: 1.0, y: 5.2, w: 8, h: 0.28,
      fontSize: 10, color: C.muted, align: "center", fontFace: "Calibri", margin: 0
    });
    s.addNotes("Closing slide. Thank evaluators. Open for questions.");
  }

  // ─── Write ───────────────────────────────────────────────────────────────────
  const outPath = "/mnt/user-data/outputs/AcadAssist_Presentation.pptx";
  await pres.writeFile({ fileName: "AcadAssist_Presentation.pptx" });
  console.log("✅  Written →", outPath);
}

build().catch(err => { console.error(err); process.exit(1); });