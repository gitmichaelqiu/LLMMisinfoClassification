from reportlab.lib import colors
from reportlab.pdfgen import canvas


OUT = "output/pdf/verification_arbitrage_mindmap.pdf"
PAGE_W, PAGE_H = 960, 600


C = {
    "bg": colors.HexColor("#fbfaf2"),
    "ink": colors.HexColor("#172033"),
    "muted": colors.HexColor("#566070"),
    "blue": colors.HexColor("#2367c9"),
    "blue2": colors.HexColor("#eaf2ff"),
    "red": colors.HexColor("#c83232"),
    "red2": colors.HexColor("#fff0f0"),
    "green": colors.HexColor("#168a52"),
    "green2": colors.HexColor("#eaf8f1"),
    "gold": colors.HexColor("#c48718"),
    "gold2": colors.HexColor("#fff7df"),
    "line": colors.HexColor("#cfd6df"),
    "white": colors.white,
}


def t(c, x, y, s, size=9, color=None, font="Helvetica", align="left"):
    c.setFillColor(color or C["ink"])
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, s)
    elif align == "right":
        c.drawRightString(x, y, s)
    else:
        c.drawString(x, y, s)


def wrap(c, text, x, y, w, size=8.5, leading=10.5, color=None, font="Helvetica"):
    words = text.split()
    line = ""
    yy = y
    c.setFont(font, size)
    c.setFillColor(color or C["ink"])
    for word in words:
        trial = (line + " " + word).strip()
        if c.stringWidth(trial, font, size) <= w:
            line = trial
        else:
            c.drawString(x, yy, line)
            yy -= leading
            line = word
    if line:
        c.drawString(x, yy, line)
    return yy - leading


def box(c, x, y, w, h, title=None, fill=None, stroke=None, title_color=None, r=6):
    c.setFillColor(fill or C["white"])
    c.setStrokeColor(stroke or C["line"])
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1)
    if title:
        t(c, x + 8, y + h - 14, title, 8.5, title_color or C["ink"], "Helvetica-Bold")


def arrow(c, x1, y1, x2, y2, color=None, width=1.1):
    col = color or C["muted"]
    c.setStrokeColor(col)
    c.setFillColor(col)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    ah = 6
    for da in (2.65, -2.65):
        c.line(x2, y2, x2 + ah * math.cos(ang + da), y2 + ah * math.sin(ang + da))


def pill(c, x, y, w, h, label, fill, stroke, color=None, size=8):
    box(c, x, y, w, h, None, fill, stroke, r=5)
    t(c, x + w / 2, y + h / 2 - 3, label, size, color or C["ink"], "Helvetica-Bold", "center")


def result_card(c, x, y, w, h, title, lines, accent=C["blue"], warn=None):
    box(c, x, y, w, h, title, C["white"], accent)
    yy = y + h - 30
    for line in lines:
        if line.startswith("Key:"):
            yy = wrap(c, line[4:].strip(), x + 9, yy, w - 18, 7.4, 9, C["muted"], "Helvetica-Bold")
        else:
            yy = wrap(c, line, x + 9, yy, w - 18, 7.2, 8.8, C["ink"])
    if warn:
        t(c, x + 9, y + 9, warn, 7.2, C["red"], "Helvetica-Bold")


def draw_architecture(c):
    x, y, w, h = 28, 232, 260, 258
    box(c, x, y, w, h, "SYSTEM ARCHITECTURE", C["blue2"], C["blue"], C["blue"])
    cx = x + 18
    pill(c, cx, y + 210, 98, 22, "Breaking News Feed", C["white"], C["blue"])
    arrow(c, cx + 98, y + 221, cx + 121, y + 221, C["blue"])
    pill(c, cx + 124, y + 210, 105, 22, "System 0 Filter", C["white"], C["blue"])
    t(c, cx + 132, y + 197, "entity check + panic keywords", 6.8, C["muted"])
    arrow(c, cx + 176, y + 210, cx + 92, y + 162, C["blue"])
    arrow(c, cx + 176, y + 210, cx + 176, y + 162, C["blue"])
    pill(c, cx + 10, y + 132, 118, 30, "A. System 1 Fast Trader", C["white"], C["blue"], size=7.6)
    t(c, cx + 20, y + 121, "FinBERT/GBDT, T0~50ms", 6.8, C["muted"])
    pill(c, cx + 138, y + 132, 116, 30, "B. System 2 LLM Verifier", C["white"], C["blue"], size=7.3)
    t(c, cx + 146, y + 121, "Single-Shot CoT + Dual RAG", 6.8, C["muted"])
    pill(c, cx + 144, y + 95, 50, 20, "Static News", C["white"], C["line"], C["muted"], 6.8)
    pill(c, cx + 200, y + 95, 50, 20, "Social Stream", C["white"], C["line"], C["muted"], 6.8)
    arrow(c, cx + 88, y + 132, cx + 127, y + 84, C["blue"])
    arrow(c, cx + 196, y + 132, cx + 151, y + 84, C["blue"])
    pill(c, cx + 92, y + 58, 92, 26, "Decision Controller", C["gold2"], C["gold"], size=7.5)
    t(c, cx + 18, y + 38, "REAL -> HOLD", 7.4, C["green"], "Helvetica-Bold")
    t(c, cx + 96, y + 38, "FAKE -> REVERSE", 7.4, C["red"], "Helvetica-Bold")
    t(c, cx + 184, y + 38, "UNCERTAIN -> HEDGE", 6.6, C["gold"], "Helvetica-Bold")
    arrow(c, cx + 184, y + 58, cx + 212, y + 24, C["blue"])
    pill(c, cx + 174, y + 6, 76, 20, "Market Execution", C["green2"], C["green"], size=7)
    c.setStrokeColor(C["line"])
    c.line(x + 20, y + 185, x + w - 20, y + 185)
    t(c, x + 22, y + 174, "T0=50ms", 7, C["blue"], "Helvetica-Bold")
    t(c, x + 104, y + 174, "T1=5s", 7, C["green"], "Helvetica-Bold")
    t(c, x + 186, y + 174, "T2=300s", 7, C["red"], "Helvetica-Bold")
    arrow(c, x + 62, y + 177, x + 100, y + 177, C["green"])
    arrow(c, x + 132, y + 177, x + 182, y + 177, C["green"])
    t(c, x + 98, y + 164, "Verification Arbitrage Window", 7.2, C["green"], "Helvetica-Bold")


def draw_economic_core(c):
    x, y, w, h = 306, 196, 302, 294
    box(c, x, y, w, h, "ECONOMIC CORE", C["white"], C["gold"], C["gold"])
    t(c, x + w / 2, y + h - 36, "Verification Delay Beats Panic at the Economic Crossover", 10.2, C["ink"], "Helvetica-Bold", "center")
    box(c, x + 16, y + 194, 128, 54, None, C["red2"], C["red"])
    t(c, x + 26, y + 230, "Trade-First", 9, C["red"], "Helvetica-Bold")
    wrap(c, "News -> immediate SELL. If fake: panic loss + market impact.", x + 26, y + 216, 108, 7.2, 9, C["ink"])
    box(c, x + 158, y + 194, 128, 54, None, C["green2"], C["green"])
    t(c, x + 168, y + 230, "Verify-First", 9, C["green"], "Helvetica-Bold")
    wrap(c, "News -> wait 5s. REAL -> hold. FAKE -> reverse/hedge.", x + 168, y + 216, 108, 7.2, 9, C["ink"])
    box(c, x + 22, y + 132, 258, 44, None, C["gold2"], C["gold"])
    t(c, x + w / 2, y + 156, "E[P&L]_Verify > E[P&L]_Trade", 12, C["ink"], "Helvetica-Bold", "center")
    t(c, x + w / 2, y + 141, "when P(fake) > crossover threshold", 8.5, C["muted"], "Helvetica-Bold", "center")
    labels = [("Empirical crossover", "~4-6%", C["green"]), ("Stress-regime crossover", "~0.3-0.5%", C["green"]), ("Panic fake-news loss", "~$30K", C["red"]), ("Real-news wait cost", "~$1.2K", C["gold"]), ("Asymmetry", "~25x", C["ink"])]
    kx = x + 18
    for i, (a, b, col) in enumerate(labels):
        xx = kx + i * 55
        t(c, xx, y + 111, b, 10, col, "Helvetica-Bold")
        wrap(c, a, xx, y + 99, 48, 6.2, 7, C["muted"])
    gx, gy, gw, gh = x + 30, y + 24, 240, 64
    c.setStrokeColor(C["line"])
    c.line(gx, gy, gx, gy + gh)
    c.line(gx, gy, gx + gw, gy)
    t(c, gx + gw / 2, gy - 12, "P(fake)", 7, C["muted"], align="center")
    t(c, gx - 17, gy + gh / 2, "Expected P&L", 7, C["muted"], align="center")
    c.setStrokeColor(C["blue"])
    c.setLineWidth(2)
    c.line(gx + 8, gy + 52, gx + gw - 10, gy + 11)
    c.setStrokeColor(C["red"])
    c.line(gx + 8, gy + 42, gx + gw - 10, gy + 34)
    ix, iy = gx + 92, gy + 36
    c.setFillColor(C["gold"])
    c.circle(ix, iy, 4, fill=1, stroke=0)
    t(c, ix + 8, iy + 4, "crossover point", 7.2, C["gold"], "Helvetica-Bold")
    t(c, gx + gw - 72, gy + 48, "Trade-First", 7, C["blue"], "Helvetica-Bold")
    t(c, gx + gw - 72, gy + 30, "Verify-First", 7, C["red"], "Helvetica-Bold")


def draw_findings(c):
    x, y, w, h = 28, 28, 580, 150
    box(c, x, y, w, h, "KEY EXPERIMENTAL FINDINGS", C["white"], C["line"])
    cw, gap = 134, 9
    result_card(c, x + 12, y + 18, cw, 104, "1. LLM Verifier",
                ["Single-Shot CoT: R 0.92, P 0.68, 2.2s",
                 "Voting N=5: R 0.80, P 0.50-0.57, 4.7s",
                 "MoA Debate: R 1.00, P 0.50, 3.5s"],
                C["blue"], "MoA degeneracy: always-FAKE bias")
    result_card(c, x + 12 + (cw + gap), y + 18, cw, 104, "2. OOD Generalization",
                ["Synthetic F1 ~= 0.99", "Human OOD F1 ~= 0.66-0.72",
                 "Key: Classical baselines collapse on human-authored rumors."], C["gold"])
    result_card(c, x + 12 + 2 * (cw + gap), y + 18, cw, 104, "3. Market Microstructure",
                ["Square-root impact:", "dP = mid * Y * sigma * sqrt(Q/V)",
                 "42.4x single-trade execution cost improvement",
                 "1.2-2.3x portfolio improvement"], C["green"])
    result_card(c, x + 12 + 3 * (cw + gap), y + 18, cw, 104, "4. Adversarial Robustness",
                ["Social-stream poisoning 0-75%",
                 "Recall remains ~= 1.0",
                 "Precision degrades, but verifier still catches fake events"], C["red"])


def draw_questions(c):
    x, y, w, h = 642, 28, 286, 462
    box(c, x, y, w, h, "DISCUSSION QUESTIONS FOR PROFESSOR", C["white"], C["line"])
    clusters = [
        ("Modeling", ["Why does MoA debate become always-FAKE?", "GPT/Claude/Gemini instead of DeepSeek?", "Binary vs HOLD/HEDGE/INTERVENE gates?"], C["blue"]),
        ("Data", ["Need real social/news API data?", "Minimum third-party OOD dataset size?", "How to create realistic adversarial content?"], C["gold"]),
        ("Finance", ["Is fake-news base rate high enough?", "Better hedge than 5% OTM puts?", "Portfolio-level correlated events?"], C["green"]),
        ("Publication", ["NLP, computational finance, or finance journal?", "Frame MoA as negative result?", "Minimal experiment to strengthen publication case?"], C["red"]),
    ]
    yy = y + h - 48
    for title, bullets, col in clusters:
        t(c, x + 14, yy, title, 9, col, "Helvetica-Bold")
        yy -= 13
        for b in bullets:
            yy = wrap(c, "- " + b, x + 18, yy, w - 34, 7.4, 9, C["ink"])
        yy -= 8


def main():
    c = canvas.Canvas(OUT, pagesize=(PAGE_W, PAGE_H))
    c.setFillColor(C["bg"])
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    t(c, PAGE_W / 2, 566, "Verification Arbitrage", 25, C["ink"], "Helvetica-Bold", "center")
    t(c, PAGE_W / 2, 544, "LLM-Assisted Fake News Mitigation for MFT", 14, C["blue"], "Helvetica-Bold", "center")
    box(c, 218, 503, 526, 30, None, C["green2"], C["green"])
    t(c, PAGE_W / 2, 514, "LLM verification at T1~5s creates a trading advantage before human verification at T2~300s.", 10.5, C["green"], "Helvetica-Bold", "center")
    t(c, PAGE_W / 2, 20, "Architecture -> Verification Delay -> Economic Crossover -> Market Impact -> Deployment Feasibility", 8.5, C["muted"], "Helvetica-Bold", "center")
    draw_architecture(c)
    draw_economic_core(c)
    draw_findings(c)
    draw_questions(c)
    c.save()


if __name__ == "__main__":
    main()
