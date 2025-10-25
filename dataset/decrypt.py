import os
import pickle
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding

def json_load(filename):
    with open(filename, "rb") as f:
        lines = []
        for line in f:
            data = json.loads(line)
            lines.append(data)
    return lines

def json_save(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

def pickle_save_encrypted(data, filename, public_key_path):
    # Serialize dataset (can be large)
    serialized_data = pickle.dumps(data)

    # Generate random AES key and IV
    aes_key = os.urandom(32)  # AES-256
    iv = os.urandom(16)

    # Encrypt the serialized data with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    # PKCS7 padding
    padding_len = 16 - (len(serialized_data) % 16)
    padded_data = serialized_data + bytes([padding_len] * padding_len)

    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Load RSA public key
    with open(public_key_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())

    # Encrypt AES key with RSA
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        rsa_padding.OAEP(
            mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Save to file: [2-byte key length][encrypted AES key][IV][encrypted data]
    with open(filename, "wb") as f:
        f.write(len(encrypted_aes_key).to_bytes(2, "big"))
        f.write(encrypted_aes_key)
        f.write(iv)
        f.write(encrypted_data)

def pickle_load_encrypted(filename, private_key_path, password=None):

    # Load private RSA key
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=password.encode() if password else None,
        )

    with open(filename, "rb") as f:
        key_len = int.from_bytes(f.read(2), "big")
        encrypted_key = f.read(key_len)
        iv = f.read(16)
        encrypted_data = f.read()

    # Decrypt AES key
    aes_key = private_key.decrypt(
        encrypted_key,
        rsa_padding.OAEP(
            mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt data
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove PKCS7 padding
    padding_len = padded_data[-1]
    data = padded_data[:-padding_len]

    return pickle.loads(data)

if __name__ == '__main__':
    train = json_load('train.json')
    dev = json_load('dev.json')
    test = json_load('test.json')

    pickle_save_encrypted(train, 'train.save', 'public_key.pem')
    pickle_save_encrypted(dev, 'dev.save', 'public_key.pem')
    pickle_save_encrypted(test, 'test.save', 'public_key.pem')

    # train = 'train.save'
    # dev = 'dev.save'
    # test = 'test.save'

    # train = pickle_load_encrypted(train, 'private_key.pem')
    # dev = pickle_load_encrypted(dev, 'private_key.pem')
    # test = pickle_load_encrypted(test, 'private_key.pem')

    # json_save(train, 'train.json')
    # json_save(dev, 'dev.json')
    # json_save(test, 'test.json')
