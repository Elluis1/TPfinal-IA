from db import create_db, insert_face, load_faces
from preprocess import image_to_vector
from hamming import HammingNetwork

def registrar_persona(nombre, path_foto):
    vec = image_to_vector(path_foto)
    insert_face(nombre, vec)
    print(vec)
    print(f"[OK] Persona {nombre} registrada en la BD")

def reconocer_persona(path_foto):
    names, vectors = load_faces()
    vec = image_to_vector(path_foto)
    network = HammingNetwork(names, vectors)
    nombre, dist = network.classify(vec)
    print(f"La foto se parece más a: {nombre} (distancia={dist})")

if __name__ == "__main__":
    create_db()

    opcion = input("1) Registrar persona\n2) Reconocer persona\nElige: ")

    if opcion == "1":
        nombre = input("Nombre de la persona: ")
        foto = input("Ruta de la foto: ")
        registrar_persona(nombre, foto)

    elif opcion == "2":
        foto = input("Ruta de la foto a reconocer: ")
        reconocer_persona(foto)
