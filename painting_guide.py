from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import Flowable

# ── Paleta ──────────────────────────────────────────────────────────────────
C_BLACK      = colors.HexColor("#0A0A0A")
C_DARK_RED   = colors.HexColor("#1A0000")
C_RED        = colors.HexColor("#8B0000")
C_GOLD       = colors.HexColor("#B8860B")
C_GOLD_LIGHT = colors.HexColor("#DAA520")
C_ORANGE     = colors.HexColor("#CC4400")
C_BLUE_COLD  = colors.HexColor("#1C3A5E")
C_GREEN_DARK = colors.HexColor("#1A2E1A")
C_GREY_GREEN = colors.HexColor("#4A5E4A")
C_WHITE      = colors.HexColor("#F5F5F0")
C_CREAM      = colors.HexColor("#F0EAD6")
C_SECTION_BG = colors.HexColor("#0F0F0F")
C_STEP_BG    = colors.HexColor("#1A1A1A")
C_BORDER     = colors.HexColor("#3A1A00")
C_ACCENT     = colors.HexColor("#CC2200")
C_GOLD_ACC   = colors.HexColor("#C9962A")
C_COLD_ACC   = colors.HexColor("#2A4A7A")

W, H = A4

# ── Estilos ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, parent='Normal', **kwargs):
    return ParagraphStyle(name, parent=styles[parent], **kwargs)

sTitle = make_style('sTitle',
    fontSize=28, textColor=C_WHITE, alignment=TA_CENTER,
    fontName='Helvetica-Bold', spaceAfter=4, leading=34)

sSubtitle = make_style('sSubtitle',
    fontSize=13, textColor=C_GOLD_LIGHT, alignment=TA_CENTER,
    fontName='Helvetica-Oblique', spaceAfter=2, leading=16)

sTagline = make_style('sTagline',
    fontSize=9, textColor=C_GREY_GREEN, alignment=TA_CENTER,
    fontName='Helvetica', spaceAfter=0, leading=12)

sSectionTitle = make_style('sSectionTitle',
    fontSize=15, textColor=C_WHITE, alignment=TA_LEFT,
    fontName='Helvetica-Bold', spaceAfter=2, leading=19,
    leftIndent=3)

sStepNum = make_style('sStepNum',
    fontSize=22, textColor=C_ACCENT, alignment=TA_CENTER,
    fontName='Helvetica-Bold', leading=26)

sStepTitle = make_style('sStepTitle',
    fontSize=11, textColor=C_GOLD_LIGHT, alignment=TA_LEFT,
    fontName='Helvetica-Bold', leading=14)

sStepBody = make_style('sStepBody',
    fontSize=8.5, textColor=C_CREAM, alignment=TA_JUSTIFY,
    fontName='Helvetica', leading=12, spaceAfter=3)

sWarning = make_style('sWarning',
    fontSize=8, textColor=colors.HexColor("#FFD700"), alignment=TA_LEFT,
    fontName='Helvetica-Bold', leading=11)

sNote = make_style('sNote',
    fontSize=8, textColor=C_GREY_GREEN, alignment=TA_LEFT,
    fontName='Helvetica-Oblique', leading=11)

sTableHeader = make_style('sTableHeader',
    fontSize=8, textColor=C_WHITE, alignment=TA_CENTER,
    fontName='Helvetica-Bold', leading=10)

sTableCell = make_style('sTableCell',
    fontSize=7.5, textColor=C_CREAM, alignment=TA_LEFT,
    fontName='Helvetica', leading=10)

sTableCellC = make_style('sTableCellC',
    fontSize=7.5, textColor=C_CREAM, alignment=TA_CENTER,
    fontName='Helvetica', leading=10)

sFooter = make_style('sFooter',
    fontSize=7, textColor=colors.HexColor("#444444"), alignment=TA_CENTER,
    fontName='Helvetica', leading=9)

sPhaseName = make_style('sPhaseName',
    fontSize=9, textColor=C_GOLD_ACC, alignment=TA_LEFT,
    fontName='Helvetica-Bold', leading=11)

sFlowItem = make_style('sFlowItem',
    fontSize=8, textColor=C_CREAM, alignment=TA_LEFT,
    fontName='Helvetica', leading=11, leftIndent=8)

sPrinciple = make_style('sPrinciple',
    fontSize=8.5, textColor=C_CREAM, alignment=TA_JUSTIFY,
    fontName='Helvetica', leading=12)

# ── Flowables personalizados ─────────────────────────────────────────────────
class ColorBar(Flowable):
    def __init__(self, colors_list, width, height=6):
        super().__init__()
        self.colors_list = colors_list
        self.width = width
        self.height = height

    def draw(self):
        n = len(self.colors_list)
        w = self.width / n
        for i, c in enumerate(self.colors_list):
            self.canv.setFillColor(c)
            self.canv.rect(i * w, 0, w, self.height, fill=1, stroke=0)

class SectionHeader(Flowable):
    def __init__(self, number, title, subtitle, width, accent_color=C_ACCENT):
        super().__init__()
        self.number = number
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.accent = accent_color
        self.height = 28

    def draw(self):
        c = self.canv
        # Fondo
        c.setFillColor(C_SECTION_BG)
        c.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=0)
        # Barra lateral
        c.setFillColor(self.accent)
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        # Número
        c.setFillColor(self.accent)
        c.setFont('Helvetica-Bold', 18)
        c.drawString(10, 7, self.number)
        # Título
        c.setFillColor(C_WHITE)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(32, 13, self.title)
        # Subtítulo
        c.setFillColor(C_GOLD_LIGHT)
        c.setFont('Helvetica-Oblique', 8)
        c.drawString(32, 4, self.subtitle)

class StepBlock(Flowable):
    """Bloque de paso con número, título y cuerpo."""
    def __init__(self, num, title, body_lines, width, notes=None, warning=None):
        super().__init__()
        self.num = num
        self.title = title
        self.body_lines = body_lines
        self.notes = notes or []
        self.warning = warning
        self.width = width
        # Calcular altura dinámica
        lines = len(body_lines)
        note_lines = len(notes) if notes else 0
        warn_lines = 1 if warning else 0
        self.height = 18 + lines * 11 + note_lines * 10 + warn_lines * 10 + 6

    def draw(self):
        c = self.canv
        c.setFillColor(C_STEP_BG)
        c.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=0)
        # Borde izquierdo
        c.setFillColor(C_GOLD_ACC)
        c.rect(0, 0, 2, self.height, fill=1, stroke=0)
        # Número
        c.setFillColor(C_ACCENT)
        c.setFont('Helvetica-Bold', 16)
        c.drawString(7, self.height - 17, self.num)
        # Título
        c.setFillColor(C_GOLD_LIGHT)
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(28, self.height - 16, self.title)
        # Línea separadora
        c.setStrokeColor(colors.HexColor("#2A2A2A"))
        c.setLineWidth(0.5)
        c.line(6, self.height - 20, self.width - 6, self.height - 20)
        # Cuerpo
        y = self.height - 31
        c.setFillColor(C_CREAM)
        c.setFont('Helvetica', 8)
        for line in self.body_lines:
            if line.startswith('→'):
                c.setFillColor(C_GOLD_ACC)
                c.setFont('Helvetica-Bold', 8)
                c.drawString(8, y, line)
                c.setFillColor(C_CREAM)
                c.setFont('Helvetica', 8)
            else:
                c.drawString(8, y, line)
            y -= 11
        # Notas
        if self.notes:
            for note in self.notes:
                c.setFillColor(C_GREY_GREEN)
                c.setFont('Helvetica-Oblique', 7.5)
                c.drawString(8, y, note)
                y -= 10
        # Warning
        if self.warning:
            c.setFillColor(colors.HexColor("#FFD700"))
            c.setFont('Helvetica-Bold', 7.5)
            c.drawString(8, y, f'⚠ {self.warning}')

class TempGradientBar(Flowable):
    """Barra de gradiente de temperatura."""
    def __init__(self, width, height=14):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        zones = [
            (C_BLUE_COLD, "FRÍO EXTREMO\nContacto suelo"),
            (C_COLD_ACC, "FRÍO\nZona baja"),
            (C_GREY_GREEN, "TRANSICIÓN\nVerde-gris"),
            (C_RED, "ROJO BASE\nZona media"),
            (C_ORANGE, "ROJO ARDIENTE\nZona alta"),
            (colors.HexColor("#FF6600"), "FUEGO\nAristas extremas"),
        ]
        n = len(zones)
        w = self.width / n
        for i, (col, label) in enumerate(zones):
            c.setFillColor(col)
            c.rect(i * w, 4, w, self.height - 4, fill=1, stroke=0)
            c.setFillColor(C_WHITE)
            c.setFont('Helvetica', 5)
            lines = label.split('\n')
            c.drawCentredString(i * w + w/2, 8, lines[0])
        # Flecha
        c.setStrokeColor(C_WHITE)
        c.setLineWidth(0.5)
        c.line(0, 2, self.width, 2)
        c.drawString(0, 0, "FRÍO")
        c.drawRightString(self.width, 0, "CALIENTE")


# ── Helpers ───────────────────────────────────────────────────────────────────
def sp(n=1):
    return Spacer(1, n * mm)

def hr(color=C_BORDER, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=2*mm)

def section_header(num, title, subtitle, doc_width, color=C_ACCENT):
    h = SectionHeader(num, title, subtitle, doc_width, color)
    return [h, sp(2)]

def paint_table(headers, rows, col_widths, doc_width):
    header_row = [Paragraph(h, sTableHeader) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([Paragraph(str(cell), sTableCell if i == 0 else sTableCellC)
                     for i, cell in enumerate(row)])
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0A00")),
        ('BACKGROUND', (0,1), (-1,-1), C_STEP_BG),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
        ('LINEBELOW', (0,0), (-1,0), 1, C_ACCENT),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t

def flow_sequence(steps):
    """Devuelve lista de párrafos con la secuencia de pasos."""
    items = []
    for i, step in enumerate(steps):
        arrow = "→ " if i < len(steps)-1 else "✓ "
        items.append(Paragraph(f"{arrow}{step}", sFlowItem))
    return items

def color_swatch_table(swatches, doc_width):
    """Tabla de swatches de color: nombre, función, color hex."""
    data = [[Paragraph("COLOR", sTableHeader),
             Paragraph("FUNCIÓN", sTableHeader),
             Paragraph("ZONA DE APLICACIÓN", sTableHeader)]]
    for name, func, zone, hex_col in swatches:
        row = [
            Paragraph(name, sTableCell),
            Paragraph(func, sTableCell),
            Paragraph(zone, sTableCell),
        ]
        data.append(row)
    cw = [doc_width*0.30, doc_width*0.25, doc_width*0.45]
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0A00")),
        ('BACKGROUND', (0,1), (-1,-1), C_STEP_BG),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
        ('LINEBELOW', (0,0), (-1,0), 1, C_ACCENT),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t


# ── COVER PAGE ───────────────────────────────────────────────────────────────
def cover_page(canvas, doc):
    canvas.saveState()
    W, H = A4
    # Fondo negro total
    canvas.setFillColor(C_BLACK)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Banda superior roja oscura
    canvas.setFillColor(C_DARK_RED)
    canvas.rect(0, H - 60*mm, W, 60*mm, fill=1, stroke=0)
    # Banda inferior dorada muy oscura
    canvas.setFillColor(colors.HexColor("#1A0E00"))
    canvas.rect(0, 0, W, 30*mm, fill=1, stroke=0)
    # Línea de separación dorada
    canvas.setStrokeColor(C_GOLD)
    canvas.setLineWidth(1.5)
    canvas.line(15*mm, H - 60*mm, W - 15*mm, H - 60*mm)
    canvas.line(15*mm, 30*mm, W - 15*mm, 30*mm)
    # Gradiente de temperatura (simulado con rectángulos)
    bar_colors = [C_BLUE_COLD, C_COLD_ACC, C_GREY_GREEN, C_RED, C_ORANGE, colors.HexColor("#FF6600")]
    bw = (W - 30*mm) / len(bar_colors)
    for i, bc in enumerate(bar_colors):
        canvas.setFillColor(bc)
        canvas.rect(15*mm + i*bw, H - 65*mm, bw, 5*mm, fill=1, stroke=0)
    # Título principal
    canvas.setFillColor(C_WHITE)
    canvas.setFont('Helvetica-Bold', 36)
    canvas.drawCentredString(W/2, H - 90*mm, "BLOOD ANGELS")
    canvas.setFillColor(C_ACCENT)
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawCentredString(W/2, H - 100*mm, "GUÍA DE PINTURA SHOWCASE")
    canvas.setFillColor(C_GOLD_LIGHT)
    canvas.setFont('Helvetica-Oblique', 11)
    canvas.drawCentredString(W/2, H - 110*mm, "Campaña de Armageddon — Protocolo Lumínico Avanzado")
    # Descripción
    canvas.setFillColor(C_CREAM)
    canvas.setFont('Helvetica', 9)
    canvas.drawCentredString(W/2, H - 125*mm, "Técnica de pincel seco multicapa · Subsurface scattering simulado")
    canvas.drawCentredString(W/2, H - 133*mm, "Control de temperatura cromática · Oclusión ambiental · Luces rebotadas")
    canvas.drawCentredString(W/2, H - 141*mm, "Sombras proyectadas · Metal híbrido TMM+Glazes")
    # Decoración central
    canvas.setStrokeColor(C_GOLD)
    canvas.setLineWidth(0.8)
    canvas.rect(15*mm, H/2 - 55*mm, W - 30*mm, 70*mm, stroke=1, fill=0)
    canvas.setFillColor(colors.HexColor("#0F0F0F"))
    canvas.rect(16*mm, H/2 - 54*mm, W - 32*mm, 68*mm, fill=1, stroke=0)
    # Texto central caja
    canvas.setFillColor(C_GREY_GREEN)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawCentredString(W/2, H/2 + 5*mm, "PRINCIPIO FUNDAMENTAL")
    canvas.setFillColor(C_CREAM)
    canvas.setFont('Helvetica-Oblique', 9)
    canvas.drawCentredString(W/2, H/2 - 5*mm, '"El rojo no es un color. Es fuego contenido.')
    canvas.drawCentredString(W/2, H/2 - 15*mm, 'El frío del suelo de Armageddon no es ausencia de luz.')
    canvas.drawCentredString(W/2, H/2 - 25*mm, 'Es la memoria de todo lo que ha ardido."')
    # Footer
    canvas.setFillColor(C_GREY_GREEN)
    canvas.setFont('Helvetica', 7)
    canvas.drawCentredString(W/2, 20*mm, "Warhammer 40,000 · Adeptus Astartes · Blood Angels · Armageddon")
    canvas.drawCentredString(W/2, 15*mm, "Guía de uso personal — No oficial — Games Workshop Ltd.")
    canvas.restoreState()

def normal_page(canvas, doc):
    canvas.saveState()
    W, H = A4
    canvas.setFillColor(colors.HexColor("#080808"))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Banda superior
    canvas.setFillColor(colors.HexColor("#0F0000"))
    canvas.rect(0, H - 12*mm, W, 12*mm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, H - 12*mm, 4, 12*mm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(10*mm, H - 8*mm, "BLOOD ANGELS — GUÍA DE PINTURA SHOWCASE — ARMAGEDDON")
    canvas.setFillColor(C_GOLD_LIGHT)
    canvas.setFont('Helvetica', 7)
    canvas.drawRightString(W - 10*mm, H - 8*mm, f"Página {doc.page}")
    # Banda inferior
    canvas.setFillColor(colors.HexColor("#0F0000"))
    canvas.rect(0, 0, W, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(C_GREY_GREEN)
    canvas.setFont('Helvetica', 6)
    canvas.drawCentredString(W/2, 3*mm, "Documento de uso personal · No oficial · Games Workshop Ltd.")
    canvas.restoreState()


# ── DOCUMENTO ────────────────────────────────────────────────────────────────
OUTPUT = "./blood_angels_painting_guide.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=15*mm, rightMargin=15*mm,
    topMargin=18*mm, bottomMargin=12*mm,
    title="Blood Angels — Guía de Pintura Showcase — Armageddon",
    author="Guía Personal de Pintura",
)

DW = doc.width  # ancho útil
story = []

# ════════════════════════════════════════════════════════════════
# PÁGINA 1: PORTADA (en blanco, se pinta en onFirstPage)
# ════════════════════════════════════════════════════════════════
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 2: INTRODUCCIÓN Y FILOSOFÍA
# ════════════════════════════════════════════════════════════════
story += section_header("I", "FILOSOFÍA Y ARQUITECTURA LUMÍNICA",
                         "El sistema de pintura de Armageddon explicado", DW)

story.append(Paragraph(
    "Esta guía documenta un protocolo de pintura showcase completo para marines Blood Angels "
    "ambientados en la Campaña de Armageddon. El sistema se basa en cuatro principios técnicos "
    "simultáneos que interactúan entre sí para crear la ilusión de una miniatura que existe "
    "dentro de un ambiente lumínico real y coherente.", sPrinciple))
story.append(sp(2))

# Tabla de principios
principles_data = [
    ["PRINCIPIO", "QUÉ ES", "POR QUÉ IMPORTA"],
    ["Temperatura cromática", "Azules fríos en zonas bajas, naranjas cálidos en zonas altas",
     "El ojo interpreta la diferencia de temperatura como volumen y tridimensionalidad"],
    ["Oclusión ambiental", "Refuerzo de negro en zonas donde la geometría bloquea la luz",
     "Crea la sensación de que la miniatura pesa y ocupa espacio físico real"],
    ["Luz rebotada", "Reflejo del suelo de ceniza en las caras que miran hacia abajo",
     "Añade asimetría lumínica que hace cada pieza de armadura individualmente creíble"],
    ["Sombra proyectada", "Oscurecimiento donde un elemento bloquea la luz sobre otro",
     "Establece relaciones espaciales entre partes: el arma existe delante del pecho"],
]
pt = Table(principles_data, colWidths=[DW*0.20, DW*0.38, DW*0.42])
pt.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0A00")),
    ('BACKGROUND', (0,1), (-1,-1), C_STEP_BG),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
    ('LINEBELOW', (0,0), (-1,0), 1, C_ACCENT),
    ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 4), ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,0), 8), ('FONTSIZE', (0,1), (-1,-1), 7.5),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('TEXTCOLOR', (0,1), (-1,-1), C_CREAM),
]))
story.append(pt)
story.append(sp(3))

# Narrativa lumínica
story += section_header("→", "NARRATIVA LUMÍNICA: ARMAGEDDON",
                         "Por qué no hay una fuente de luz dominante", DW, C_GOLD_ACC)
story.append(Paragraph(
    "Armageddon es un mundo moribundo. El sol está oscurecido por décadas de humo industrial "
    "y la ceniza de millones de cadáveres. Los fuegos de las fábricas destruidas, los vehículos "
    "ardiendo y las explosiones constantes generan una iluminación omnidireccional de temperatura "
    "cálida-naranja. El suelo de ceniza y metal frío refleja una luz difusa fría desde abajo. "
    "Esta dualidad — fuego cálido desde todos lados, frío desde el suelo — es lo que justifica "
    "técnicamente tu esquema cromático y lo que hace que la miniatura cuente una historia "
    "sin necesitar una sola palabra de descripción.", sPrinciple))
story.append(sp(2))

# Barra de temperatura
story.append(Paragraph("MAPA DE TEMPERATURA A LO LARGO DE LA MINIATURA", sPhaseName))
story.append(sp(1))
story.append(TempGradientBar(DW, 16))
story.append(sp(3))

# Lista de materiales
story += section_header("II", "MATERIALES NECESARIOS",
                         "Inventario completo antes de empezar", DW)

mats = [
    ["PRODUCTO", "MARCA", "USO EN ESTA GUÍA"],
    ["Imprimación negra spray/aerógrafo", "Cualquier marca", "Base absoluta de toda la miniatura"],
    ["Azul Ultramarine", "Vallejo Game Color", "1er pincel seco frío — zona baja"],
    ["Azul Mágico", "Vallejo Game Color", "2o pincel seco frío — zona baja concentrada"],
    ["Toxic Mist", "Army Painter Warpaints", "3er pincel seco frío — aristas bajas y rebote suelo"],
    ["Troll Slayer Orange", "Citadel", "1er pincel seco cálido — zonas medias y altas"],
    ["Naranja Fuego", "Vallejo Game Color", "2o pincel seco cálido — aristas y puntos de luz"],
    ["Naranja Fénix Intenso", "Vallejo Xpress Color", "Intensificador localizado en picos de luz"],
    ["Contrast Blood Angels Red", "Citadel Contrast", "Filtro rojo transparente — unifica todo"],
    ["Contrast Medium", "Citadel", "Diluir el Contrast para aplicación controlada"],
    ["Oilbrusher Mecha Dark Green", "MIG", "Recesos — complementario frío del rojo"],
    ["White spirit sin olor", "Droguería", "Limpieza del oilbrusher"],
    ["Abaddon Black", "Citadel Base", "Refuerzo oclusión ambiental y sombras proyectadas"],
    ["Nuln Oil", "Citadel Shade", "Wash para sombras y base de metales"],
    ["Agrax Earthshade", "Citadel Shade", "Wash cálido para base del dorado"],
    ["Leadbelcher o Dark Steel", "Citadel / Vallejo Metal Color", "Base metálica del dorado"],
    ["Gehenna's Gold", "Citadel", "Primera luz del dorado"],
    ["Auric Armour Gold", "Citadel", "Segunda luz del dorado — aristas"],
    ["Liberator Gold", "Citadel", "Tercera luz — aristas extremas"],
    ["Stormhost Silver", "Citadel", "Punto especular máximo del dorado"],
    ["Contrast Aethermatic Blue", "Citadel Contrast", "Glaze frío para sombras del dorado"],
    ["Contrast Iyanden Yellow", "Citadel Contrast", "Glaze cálido para zonas medias del dorado"],
    ["Mechanicus Standard Grey", "Citadel Base", "Base para zonas negras"],
    ["Coelia Greenshade", "Citadel Shade", "Wash verde oscuro para zonas negras"],
    ["Dark Reaper", "Citadel Layer", "1a luz zonas negras"],
    ["Administratum Grey + Elysian Green", "Citadel", "2a luz zonas negras — gris verdoso"],
]
mt = Table(mats, colWidths=[DW*0.42, DW*0.25, DW*0.33])
mt.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0A00")),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
    ('LINEBELOW', (0,0), (-1,0), 1, C_ACCENT),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,0), 7.5), ('FONTSIZE', (0,1), (-1,-1), 7),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('TEXTCOLOR', (0,1), (-1,-1), C_CREAM),
    ('TEXTCOLOR', (0,1), (1,-1), colors.HexColor("#AAAAAA")),
]))
story.append(mt)
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 3: ROJO — FASE 1 PREPARACIÓN
# ════════════════════════════════════════════════════════════════
story += section_header("III", "ARMADURA ROJA — FASE 1: PREPARACIÓN",
                         "Imprimación, oclusión ambiental y base térmica", DW)

steps_prep = [
    ("01", "IMPRIMACIÓN NEGRA",
     ["Imprime toda la miniatura en negro puro.",
      "→ Spray a 25-30 cm de distancia, en capas finas.",
      "→ Dos pasadas: una cenital, una lateral.",
      "→ Objetivo: negro uniforme sin empastes.",
      "El negro es la sombra más profunda. Todo lo que no toques",
      "después quedará en negro absoluto.",],
     ["Espera 30 min mínimo antes de continuar."],
     None),
    ("02", "REFUERZO DE OCLUSIÓN AMBIENTAL",
     ["Identifica las zonas de oclusión geométrica:",
      "→ Axilas y cara interior de los brazos.",
      "→ Interior de rodillas y codos.",
      "→ Zona entre mochila/dorsal y espalda.",
      "→ Bajo el pecho si hay voladizo de armadura.",
      "→ Entre dedos y palma de la mano.",
      "Aplica Abaddon Black muy diluido (1:3 con agua) con",
      "pincel fino solo en esas zonas. NO en superficie general.",
      "Estas zonas quedarán negro absoluto permanente.",],
     ["La oclusión ambiental es ausencia total de luz,",
      "no sombra. Son cosas distintas."],
     None),
    ("03", "PREPARACIÓN DEL PINCEL SECO",
     ["Carga el pincel seco con el color indicado.",
      "→ Carga abundante en el pincel.",
      "→ Elimina el 95% de la pintura en papel de cocina.",
      "→ El pincel debe dejar apenas una huella al frotarlo.",
      "Cada color se aplica en 3 pasadas del mismo tono:",
      "→ Pasada 1: zona amplia, presión muy suave.",
      "→ Pasada 2: zona más pequeña, misma presión.",
      "→ Pasada 3: solo aristas y puntos más altos.",
      "Las 3 pasadas son del MISMO color, no distintos.",],
     ["El objetivo es construir volumen, no manchar."],
     "No recargar el pincel entre pasadas del mismo color."),
]

for num, title, body, notes, warning in steps_prep:
    block = StepBlock(num, title, body, DW, notes, warning)
    story.append(KeepTogether([block, sp(2)]))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 4: ROJO — FASE 2 TEMPERATURA
# ════════════════════════════════════════════════════════════════
story += section_header("IV", "ARMADURA ROJA — FASE 2: TEMPERATURA",
                         "Construcción de la arquitectura cromática fría y cálida", DW)

steps_temp = [
    ("04", "PINCEL SECO AZUL ULTRAMARINE — ZONA BAJA",
     ["Color: Vallejo Game Color Azul Ultramarine.",
      "Zona de aplicación: tercio inferior de la miniatura.",
      "→ Pasada 1: desde la base hasta 1/3 de altura, muy suave.",
      "→ Pasada 2: hasta 1/4 de altura, más concentrado.",
      "→ Pasada 3: solo los bordes del contacto con el suelo.",
      "Simula el reflejo frío del suelo de ceniza de Armageddon.",
      "No debe subir más de un tercio de la miniatura.",],
     ["Si ves demasiado azul, has cargado demasiado el pincel."], None),
    ("05", "PINCEL SECO AZUL MÁGICO — ZONA BAJA CONCENTRADA",
     ["Color: Vallejo Game Color Azul Mágico.",
      "Zona: más pequeña que el Ultramarine. Mismo tercio inferior.",
      "→ Pasada 1: hasta 1/4 de altura.",
      "→ Pasada 2: hasta 1/5 de altura.",
      "→ Pasada 3: aristas de la zona más baja.",
      "El Azul Mágico es más claro que el Ultramarine.",
      "Construye la transición hacia el frío más intenso.",],
     None, None),
    ("06", "PINCEL SECO TOXIC MIST — ARISTAS BAJAS Y REBOTE",
     ["Color: Army Painter Warpaints Toxic Mist (turquesa-verde).",
      "Zona: MUY concentrada. Solo aristas y bordes en la base.",
      "→ Pasada 1: aristas inferiores, muy suave.",
      "→ Pasada 2: vértices inferiores más expuestos.",
      "→ Pasada 3: punto más bajo de cada elemento.",
      "ADEMÁS: aplica en caras que miran hacia abajo:",
      "→ Cara inferior de hombreras.",
      "→ Cara inferior de palmas y puños.",
      "→ Cara inferior del pecho si hay voladizo.",
      "→ Cara inferior del cañón de armas.",
      "El Toxic Mist es cian-verde: máximo complementario del rojo.",],
     ["La zona de transición entre Toxic Mist y Troll Slayer",
      "quedará sucia antes del Contrast. Es normal. No corrijas."],
     "No subas el Toxic Mist más allá del primer cuarto de la miniatura."),
    ("07", "PINCEL SECO TROLL SLAYER ORANGE — MASA PRINCIPAL",
     ["Color: Citadel Troll Slayer Orange.",
      "Zona: dos tercios superiores de la miniatura.",
      "→ Pasada 1: desde 1/3 de altura hasta arriba, amplia.",
      "→ Pasada 2: mitad superior, más concentrada.",
      "→ Pasada 3: tercio superior y aristas visibles.",
      "Este es el color base de la luz cálida de Armageddon.",
      "El Contrast Blood Angels encima lo convertirá en rojo sangre.",
      "No te preocupes si ahora parece demasiado naranja.",],
     None, None),
    ("08", "PINCEL SECO NARANJA FUEGO VGC — LUCES",
     ["Color: Vallejo Game Color Naranja Fuego.",
      "Zona: tercio superior y aristas de la mitad superior.",
      "→ Pasada 1: tercio superior, suave.",
      "→ Pasada 2: aristas del tercio superior.",
      "→ Pasada 3: vértices y puntos más altos de cada superficie.",
      "El Naranja Fuego es más amarillento que el Troll Slayer.",
      "Bajo el Contrast se convertirá en rojo ardiente casi naranja.",],
     None, None),
    ("09", "XPRESS NARANJA FÉNIX — INTENSIFICADOR DE PICOS",
     ["Color: Vallejo Xpress Color Naranja Fénix Intenso.",
      "Herramienta: pincel fino, NO pincel seco.",
      "Zona: SOLO superficies amplias del tercio superior.",
      "→ Aplica con pincel fino como una capa semitransparente.",
      "→ Solo donde quieres el rojo más ardiente y explosivo.",
      "→ NO en aristas ni recesos.",
      "El Xpress densifica el naranja localizado.",
      "Bajo el Contrast producirá el rojo más intenso de la miniatura.",],
     ["Espera a que seque el Xpress antes de aplicar el Contrast.",
      "Mínimo 15 minutos."],
     "No apliques en toda la miniatura, solo en zonas altas amplias."),
]

for num, title, body, notes, warning in steps_temp:
    block = StepBlock(num, title, body, DW, notes, warning)
    story.append(KeepTogether([block, sp(2)]))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 5: ROJO — FASE 3 CONTRAST Y ACABADO
# ════════════════════════════════════════════════════════════════
story += section_header("V", "ARMADURA ROJA — FASE 3: CONTRAST Y ACABADO",
                         "El filtro rojo, las sombras proyectadas y el oilbrusher", DW)

steps_contrast = [
    ("10", "CONTRAST BLOOD ANGELS RED — EL FILTRO",
     ["Este paso transforma toda la arquitectura previa en rojo.",
      "Mezcla: 70% Contrast Blood Angels + 30% Contrast Medium.",
      "→ Aplica con pincel suave de pelo sintético.",
      "→ Una sola capa. Deja fluir por gravedad hacia los recesos.",
      "→ No retoques zonas que ya has tocado mientras está húmedo.",
      "→ Trabaja por secciones: hombrera, brazo, pecho, pierna.",
      "Lo que verás zona a zona:",
      "→ Recesos negros: rojo oscuro casi negro.",
      "→ Zona baja azul+naranja: granate frío violáceo.",
      "→ Zona media Troll Slayer: rojo sangre saturado.",
      "→ Zona alta Naranja Fuego: rojo ardiente.",
      "→ Picos con Xpress: rojo-naranja casi fuego.",],
     ["El Contrast Medium diluido evita que el Contrast",
      "quede demasiado oscuro en zonas medias."],
     "No apliques en capas gruesas. Una capa fina es siempre mejor."),
    ("11", "SOMBRAS PROYECTADAS — PINCEL FINO",
     ["Este paso añade la relación espacial entre elementos.",
      "Mezcla: Abaddon Black + Contrast Medium (ratio 1:4).",
      "Herramienta: pincel fino de punta.",
      "Zonas donde aplicar:",
      "→ Bajo la hombrera, sobre el brazo superior.",
      "→ Bajo el cañón del arma, sobre la mano.",
      "→ Bajo la cabeza/casco, sobre el pecho.",
      "→ Donde el brazo cruza frente al torso.",
      "Técnica: línea degradada, más oscura en el borde del",
      "objeto que proyecta, más clara alejándose.",
      "En Armageddon la luz es difusa: NUNCA bordes duros.",
      "Siempre gradiente suave de oscuro a claro.",],
     ["Practica el degradado en papel antes de la miniatura.",
      "El error más común es hacerlo demasiado oscuro y duro."],
     None),
    ("12", "OILBRUSHER MECHA DARK GREEN — RECESOS",
     ["Este paso añade el complementario frío en los recesos.",
      "→ Aplica el oilbrusher directamente en la punta del receso.",
      "→ Solo en recesos, NO en superficies.",
      "→ Deja 10-15 minutos hasta que pierda el brillo húmedo.",
      "→ Limpia el exceso con pincel plano seco hacia las zonas altas.",
      "→ Movimiento siempre desde el receso hacia fuera.",
      "→ Espera 24 horas de secado completo antes de continuar.",
      "El verde oscuro en recesos es el complementario del rojo.",
      "Crea contraste simultáneo de valor Y de temperatura.",
      "Unifica con los azules fríos de la base creando coherencia.",],
     ["24 horas de secado es mínimo. El óleo tarda."],
     "No barnices hasta que el óleo esté completamente seco."),
]

for num, title, body, notes, warning in steps_contrast:
    block = StepBlock(num, title, body, DW, notes, warning)
    story.append(KeepTogether([block, sp(2)]))

# Tabla resumen de resultado por zona
story.append(sp(2))
story.append(Paragraph("RESULTADO ESPERADO POR ZONA — ARMADURA ROJA", sPhaseName))
story.append(sp(1))
result_table = [
    ["ZONA", "COLORES BASE", "RESULTADO TRAS CONTRAST", "TEMP."],
    ["Recesos profundos", "Negro puro", "Rojo casi negro", "—"],
    ["Contacto con suelo", "Negro + Ultramarine", "Granate frío oscuro", "❄❄"],
    ["Zona baja", "Ultramarine + Azul Mágico", "Granate violáceo frío", "❄"],
    ["Transición", "Toxic Mist + inicio Troll Slayer", "Granate rojizo oscuro", "~"],
    ["Zona media", "Troll Slayer Orange", "Rojo sangre saturado", "🔥"],
    ["Zona alta", "Troll Slayer + Naranja Fuego", "Rojo ardiente luminoso", "🔥🔥"],
    ["Picos y aristas", "Naranja Fuego + Xpress Fénix", "Rojo-naranja fuego vivo", "🔥🔥🔥"],
    ["Caras mirando abajo", "Toxic Mist (rebote suelo)", "Rojo frío — ilusión rebote", "❄"],
]
rt = Table(result_table, colWidths=[DW*0.20, DW*0.28, DW*0.36, DW*0.16])
rt.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0A00")),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
    ('LINEBELOW', (0,0), (-1,0), 1, C_ACCENT),
    ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,0), 7.5), ('FONTSIZE', (0,1), (-1,-1), 7.5),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('TEXTCOLOR', (0,1), (-1,-1), C_CREAM),
    ('ALIGN', (2,0), (3,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(rt)
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 6: DORADO HÍBRIDO
# ════════════════════════════════════════════════════════════════
story += section_header("VI", "DORADO HÍBRIDO — BRONCE OSCURO CON DESTELLOS",
                         "TMM real + glazes de temperatura. Sucio, ardiente, épico.", DW, C_GOLD_ACC)

story.append(Paragraph(
    "El dorado de los Blood Angels en Armageddon no es el dorado ceremonial del palacio. "
    "Es un dorado cubierto de polvo de ceniza, oscurecido por el humo, que solo destella "
    "cuando la luz de un incendio lo roza en una arista. La base es casi bronce negro. "
    "Los destellos son casi plateados. El contraste entre ambos es donde vive la magia.", sPrinciple))
story.append(sp(2))

steps_gold = [
    ("13", "BASE METÁLICA OSCURA",
     ["Color base: Citadel Leadbelcher o Vallejo Metal Color Dark Steel.",
      "→ Aplica con pincel en todas las zonas doradas.",
      "→ Cobertura completa, no importa si no es perfecta.",
      "Wash: Agrax Earthshade + Nuln Oil mezclados (ratio 2:1).",
      "→ Capa completa sobre toda la zona metálica.",
      "→ Deja fluir a los recesos.",
      "→ Espera secado completo: 30-45 minutos.",
      "Resultado: metal viejo, sucio, casi negro. Es correcto.",],
     ["El punto de partida oscuro es lo que hace que los",
      "destellos finales sean tan impactantes."], None),
    ("14", "CONSTRUCCIÓN DE LUZ METÁLICA — 4 ESCALONES",
     ["L1 — Gehenna's Gold: superficies amplias iluminadas.",
      "→ Pincel seco suave o capa muy diluida.",
      "→ Solo en superficies que recibirían luz ambiental.",
      "L2 — Auric Armour Gold: aristas y bordes.",
      "→ Pincel fino. Solo líneas de luz en aristas.",
      "L3 — Liberator Gold: aristas extremas.",
      "→ Pincel fino. Zona más pequeña que L2.",
      "L4 — Stormhost Silver: vértice de máxima luz.",
      "→ Puntual. Solo el punto más expuesto de cada arista.",
      "→ El plateado en el pico no es error: simula la",
      "   especularidad real del metal bajo luz intensa.",],
     ["Cada escalón cubre una zona más pequeña que el anterior.",
      "Si los escalones tienen el mismo tamaño, no hay contraste."],
     None),
    ("15", "GLAZES DE TEMPERATURA SOBRE EL METAL",
     ["Este paso modifica la temperatura del metal por zonas.",
      "Glaze FRÍO — Contrast Aethermatic Blue muy diluido (1:3):",
      "→ Solo en recesos profundos y zonas de oclusión.",
      "→ Convierte las sombras de marrón a negro azulado frío.",
      "→ Coherente con los azules fríos del rojo.",
      "Glaze CÁLIDO — Contrast Iyanden Yellow muy diluido (1:4):",
      "→ Solo en superficies medias entre sombra y luz.",
      "→ Calienta el bronce hacia dorado sin perder la base oscura.",
      "SIN GLAZE en las aristas de luz:",
      "→ Las luces metálicas quedan intactas.",
      "→ El contraste frío/metal puro es el máximo posible.",],
     ["Diluye mucho. Un glaze metálico demasiado denso",
      "destruye el trabajo de luz previo en segundos."],
     "Aplica el glaze frío PRIMERO. Espera que seque. Luego el cálido."),
    ("16", "OILBRUSHER EN RECESOS DEL DORADO",
     ["Color: Oilbrusher Mecha Dark Green (el mismo del rojo).",
      "→ Solo en recesos MUY profundos del dorado.",
      "→ Cantidad mucho menor que en el rojo.",
      "→ Limpieza más agresiva con white spirit.",
      "El verde del oilbrusher en los recesos del dorado",
      "se alinea cromáticamente con los recesos del rojo.",
      "Toda la miniatura comparte el mismo ambiente de sombra.",
      "Eso es coherencia lumínica global.",],
     ["24 horas de secado antes de continuar."],
     None),
]

for num, title, body, notes, warning in steps_gold:
    block = StepBlock(num, title, body, DW, notes, warning)
    story.append(KeepTogether([block, sp(2)]))

# Tabla coherencia dorado-rojo
story.append(sp(1))
story.append(Paragraph("COHERENCIA CROMÁTICA ENTRE ROJO Y DORADO", sPhaseName))
story.append(sp(1))
coh_table = [
    ["ZONA", "ROJO", "DORADO", "ELEMENTO COMPARTIDO"],
    ["Recesos profundos", "Rojo oscuro casi negro", "Bronce negro", "Oilbrusher verde oscuro"],
    ["Sombra media", "Granate frío", "Bronce con glaze azul", "Temperatura fría"],
    ["Zona media", "Rojo sangre", "Bronce dorado con glaze cálido", "Temperatura neutra-cálida"],
    ["Aristas de luz", "Rojo ardiente naranja", "Destello dorado casi plateado", "Máximo contraste"],
]
ct = Table(coh_table, colWidths=[DW*0.18, DW*0.24, DW*0.28, DW*0.30])
ct.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0E00")),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
    ('LINEBELOW', (0,0), (-1,0), 1, C_GOLD_ACC),
    ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,0), 7.5), ('FONTSIZE', (0,1), (-1,-1), 7.5),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('TEXTCOLOR', (0,1), (-1,-1), C_CREAM),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(ct)
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 7: NEGRO — JUNTAS Y ZONAS OSCURAS
# ════════════════════════════════════════════════════════════════
story += section_header("VII", "JUNTAS Y ZONAS NEGRAS DE ARMADURA",
                         "Negro oscuro con highlight gris verdoso — coherente con el oilbrusher", DW)

story.append(Paragraph(
    "Las juntas, sellos de armadura y zonas negras no son negro puro. Son un negro muy oscuro "
    "con un alma verde-gris que conecta visualmente con el oilbrusher Mecha Dark Green de los "
    "recesos. La coherencia cromática se construye usando el mismo tono de referencia "
    "en distintas zonas de la miniatura.", sPrinciple))
story.append(sp(2))

steps_black = [
    ("17", "BASE DE LAS ZONAS NEGRAS",
     ["Las juntas ya están en negro de imprimación.",
      "En zonas negras amplias (sellos de armadura, guantes, botas):",
      "→ Aplica Mechanicus Standard Grey muy diluido (1:2).",
      "→ Solo en las superficies amplias, NO en recesos.",
      "→ Una capa fina que deja el negro visible debajo.",
      "Esto crea una base ligeramente más clara que el negro puro",
      "sobre la que las luces posteriores tendrán más impacto.",],
     None, None),
    ("18", "WASH CON TONO VERDE OSCURO",
     ["Color: Citadel Coelia Greenshade.",
      "→ Capa completa sobre toda la zona negra.",
      "→ Deja fluir a los recesos.",
      "→ El Coelia Greenshade es un wash verde-negro muy oscuro.",
      "→ Unifica la base y añade el alma verde que buscamos.",
      "Resultado: negro con una sutilísima temperatura verde-fría.",
      "Apenas visible directamente, pero que el ojo procesa",
      "subconscientemente como coherente con el oilbrusher.",],
     ["Espera secado completo: 20-30 minutos."], None),
    ("19", "PRIMERA LUZ — DARK REAPER",
     ["Color: Citadel Dark Reaper.",
      "→ Pincel fino en aristas y bordes de zonas negras.",
      "→ Solo líneas de luz, no superficies amplias.",
      "→ El Dark Reaper es un gris-azul muy oscuro.",
      "→ A esta escala parece casi negro pero añade dimensión.",
      "Zonas de aplicación:",
      "→ Aristas de sellos de armadura.",
      "→ Bordes de guantes y botas.",
      "→ Nudillos y aristas de dedos.",
      "→ Juntas de armadura expuestas a la luz.",],
     None, None),
    ("20", "SEGUNDA LUZ — GRIS VERDOSO EN ARISTAS EXTREMAS",
     ["Mezcla: Administratum Grey + Elysian Green (ratio 3:1).",
      "→ Pincel fino de punta. Zona MUY pequeña.",
      "→ Solo vértices y puntos de máxima luz.",
      "→ El toque de Elysian Green lleva el gris hacia el verde.",
      "→ Este verde-gris es el mismo tono del oilbrusher aclarado.",
      "Resultado: las aristas más altas del negro tienen un",
      "destello gris-verdoso que repite el complementario frío",
      "del rojo en otro elemento de la miniatura.",
      "La coherencia cromática se hace visible a nivel global.",],
     ["Menos es más. Si lo ves claramente desde 30 cm,",
      "has puesto demasiado."],
     None),
]

for num, title, body, notes, warning in steps_black:
    block = StepBlock(num, title, body, DW, notes, warning)
    story.append(KeepTogether([block, sp(2)]))

story.append(sp(2))

# Tabla coherencia global
story += section_header("VIII", "COHERENCIA CROMÁTICA GLOBAL",
                         "Cómo los tres elementos hablan el mismo idioma visual", DW, C_GREY_GREEN)

story.append(Paragraph(
    "Una miniatura showcase no es la suma de tres elementos bien pintados. "
    "Es un único objeto con un sistema cromático coherente donde cada zona "
    "referencia a las demás. En esta miniatura el elemento que une todo es el "
    "verde oscuro del oilbrusher Mecha Dark Green.", sPrinciple))
story.append(sp(2))

global_table = [
    ["ELEMENTO", "SOMBRAS", "ZONA MEDIA", "LUCES", "NEXO CROMÁTICO"],
    ["Armadura roja",
     "Negro + verde\noilbrusher",
     "Rojo sangre\nsaturado",
     "Rojo-naranja\nardiente",
     "Verde oilbrusher\nen recesos"],
    ["Dorado híbrido",
     "Bronce negro +\nverde oilbrusher",
     "Bronce dorado\ncon glaze cálido",
     "Destello casi\nplateado",
     "Verde oilbrusher\nen recesos"],
    ["Zonas negras",
     "Negro + Coelia\nGreenshade",
     "Gris oscuro\ncon alma verde",
     "Gris-verdoso\nen aristas",
     "Gris verdoso =\noilbrusher aclarado"],
    ["Toda la miniatura",
     "Azul frío +\nverde oscuro",
     "Temperatura\nneutra",
     "Naranja-dorado\nardiente",
     "El verde es el hilo\nque une todo"],
]
gt = Table(global_table, colWidths=[DW*0.18, DW*0.20, DW*0.20, DW*0.20, DW*0.22])
gt.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0A1A0A")),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
    ('LINEBELOW', (0,0), (-1,0), 1, C_GREY_GREEN),
    ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,0), 7.5), ('FONTSIZE', (0,1), (-1,-1), 7.5),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('TEXTCOLOR', (0,1), (-1,-1), C_CREAM),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('BACKGROUND', (-1,1), (-1,-1), colors.HexColor("#0A1A0A")),
    ('TEXTCOLOR', (-1,1), (-1,-1), C_GREY_GREEN),
    ('FONTNAME', (-1,1), (-1,-1), 'Helvetica-Bold'),
]))
story.append(gt)
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# PÁGINA 8: FLUJO MAESTRO Y CHECKLIST
# ════════════════════════════════════════════════════════════════
story += section_header("IX", "FLUJO MAESTRO COMPLETO",
                         "Los 20 pasos en orden. La secuencia que no debes alterar.", DW)

all_steps = [
    "Imprimación negra completa — dos pasadas",
    "Refuerzo oclusión ambiental — Abaddon Black diluido en recesos geométricos",
    "Pincel seco Azul Ultramarine — tercio inferior, zona amplia, 3 pasadas",
    "Pincel seco Azul Mágico — zona baja concentrada, 3 pasadas",
    "Pincel seco Toxic Mist — aristas bajas + caras mirando al suelo, 3 pasadas finas",
    "Pincel seco Troll Slayer Orange — dos tercios superiores, 3 pasadas",
    "Pincel seco Naranja Fuego VGC — tercio superior y aristas altas, 3 pasadas",
    "Xpress Naranja Fénix — pincel fino solo en superficies altas amplias",
    "Contrast Blood Angels Red diluido (70/30) — capa fina, dejar fluir",
    "Sombras proyectadas — Abaddon Black 1:4 Contrast Medium, pincel fino, degradado suave",
    "Oilbrusher Mecha Dark Green — solo recesos del rojo — ESPERAR 24H",
    "Base metálica Leadbelcher en zonas doradas",
    "Wash Agrax Earthshade + Nuln Oil (2:1) en dorado",
    "Luz L1 Gehenna's Gold — superficies amplias del dorado",
    "Luz L2 Auric Armour Gold — aristas del dorado",
    "Luz L3 Liberator Gold — aristas extremas del dorado",
    "Luz L4 Stormhost Silver — vértice máximo del dorado",
    "Glaze Aethermatic Blue diluido — recesos del dorado",
    "Glaze Iyanden Yellow diluido — zonas medias del dorado",
    "Oilbrusher Mecha Dark Green — recesos profundos del dorado — ESPERAR 24H",
    "Base Mechanicus Standard Grey diluido — superficies negras amplias",
    "Wash Coelia Greenshade — zonas negras completas",
    "Luz Dark Reaper — aristas de zonas negras",
    "Luz gris-verdoso (Administratum Grey + Elysian Green 3:1) — vértices negros",
    "Barniz mate general — proteger la miniatura",
    "Barniz satinado selectivo — solo en metales, para recuperar brillo",
]

flow_data = [["#", "PASO", "ESPERA"]]
waits = {10: "24h", 19: "24h", 24: "—", 25: "—"}
for i, step in enumerate(all_steps):
    wait = waits.get(i, "—")
    if i in [10, 19]:
        wait = "24h ⚠"
    flow_data.append([
        str(i+1).zfill(2),
        step,
        wait
    ])

ft = Table(flow_data, colWidths=[DW*0.06, DW*0.82, DW*0.12])
ft.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A0A00")),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_STEP_BG, colors.HexColor("#141414")]),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor("#2A2A2A")),
    ('LINEBELOW', (0,0), (-1,0), 1, C_ACCENT),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,0), 7.5), ('FONTSIZE', (0,1), (-1,-1), 7),
    ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
    ('TEXTCOLOR', (0,1), (-1,-1), C_CREAM),
    ('TEXTCOLOR', (0,0), (0,-1), C_GOLD_ACC),
    ('ALIGN', (0,0), (0,-1), 'CENTER'),
    ('ALIGN', (2,0), (2,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    # Marcar pasos de óleo
    ('BACKGROUND', (0,11), (-1,11), colors.HexColor("#0A0A1A")),
    ('BACKGROUND', (0,20), (-1,20), colors.HexColor("#0A0A1A")),
    ('TEXTCOLOR', (2,11), (2,11), colors.HexColor("#FFD700")),
    ('TEXTCOLOR', (2,20), (2,20), colors.HexColor("#FFD700")),
]))
story.append(ft)
story.append(sp(3))

# Nota final
story.append(Paragraph(
    "⚠  Los pasos marcados con 24h son obligatorios. El óleo húmedo bajo barniz "
    "o pintura acrílica produce efecto 'frosted' irreversible. La paciencia en estos "
    "dos puntos determina si la miniatura es showcase o un accidente.", sWarning))

story.append(sp(2))
story.append(hr(C_GOLD_ACC))
story.append(sp(1))
story.append(Paragraph(
    "Esta guía ha sido generada como documento de referencia personal para el proyecto "
    "Blood Angels — Campaña de Armageddon. Todos los nombres de productos, facciones y "
    "universo son propiedad de Games Workshop Limited. Uso estrictamente personal.", sFooter))


# ── BUILD ─────────────────────────────────────────────────────────────────────
doc.build(
    story,
    onFirstPage=cover_page,
    onLaterPages=normal_page,
)
print("PDF generado correctamente.")