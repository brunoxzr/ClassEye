import os
import xml.etree.ElementTree as ET

# Caminho para os arquivos XML
xml_folder = "labels/xml"
yolo_folder = "labels/train"

# Dicionário de classes
classes = ["dormindo", "acordado", "copiando", "atento", "distraido"]  # Substitua pelas suas classes

# Criar pasta YOLO se não existir
os.makedirs(yolo_folder, exist_ok=True)

# Função para converter
def convert_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Tamanho da imagem
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    # Lista para armazenar rótulos
    yolo_annotations = []

    # Iterar pelos objetos no XML
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Calcular coordenadas normalizadas para YOLO
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Adicionar ao rótulo YOLO
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Salvar em arquivo .txt
    yolo_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")
    with open(yolo_file, "w") as f:
        f.write("\n".join(yolo_annotations))

# Converter todos os arquivos XML
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        convert_to_yolo(os.path.join(xml_folder, xml_file), yolo_folder)

print("Conversão concluída!")
