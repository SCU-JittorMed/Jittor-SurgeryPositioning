import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.app import PositioningApp

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    app = PositioningApp(config)
    app.setup()
    
    print("Select mode:")
    print("s: Shared Memory")
    print("t: TCP Socket (default)")
    print("n: No IPC")
    print("m: Manual configuration")
    
    choice = input("Choice: ").strip().lower()
    
    show_image = True
    enable_ipc = True
    ipc_method = 'tcp_socket'
    
    if choice == 's':
        ipc_method = 'shared_memory'
    elif choice == 't':
        ipc_method = 'tcp_socket'
    elif choice == 'n':
        enable_ipc = False
    elif choice == 'm':
        show_img_in = input("Show Image? (y/n, default y): ").strip().lower()
        show_image = show_img_in != 'n'
        ipc_in = input("Enable IPC? (y/n, default y): ").strip().lower()
        enable_ipc = ipc_in != 'n'
        if enable_ipc:
            print("IPC Method: 1. Shared Memory, 2. TCP Socket")
            m_in = input("Select (1/2): ").strip()
            if m_in == '1':
                ipc_method = 'shared_memory'
            else:
                ipc_method = 'tcp_socket'
    
    app.run(show_image=show_image, enable_ipc=enable_ipc, ipc_method=ipc_method)

if __name__ == '__main__':
    main()
