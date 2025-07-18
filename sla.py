import subprocess

def convert_h264_to_mp4(input_file, output_file):
    """
    Converte um vídeo codificado em H.264 para o formato MP4 usando FFmpeg.
    
    :param input_file: Caminho do arquivo de entrada (.h264)
    :param output_file: Caminho do arquivo de saída (.mp4)
    """
    command = [
        "ffmpeg",
        "-framerate", "30",  # Ajuste a taxa de quadros conforme necessário
        "-i", input_file,
        "-c:v", "copy",  # Copia o vídeo sem recodificar
        "-movflags", "faststart",  # Otimiza para reprodução na web
        output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversão concluída: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao converter o vídeo: {e}")

# Exemplo de uso
input_video = "ceep.h264"  # Substitua pelo seu arquivo de entrada
output_video = "video.mp4"  # Nome do arquivo de saída

convert_h264_to_mp4(input_video, output_video)
