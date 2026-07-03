from __future__ import annotations

from typing import Iterable


def write_holding_registers(
    host: str,
    port: int,
    start_address: int,
    values: Iterable[int],
    unit_id: int = 1,
) -> None:
    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency pymodbus. Install requirements first: python -m pip install -r requirements.txt"
        ) from exc

    client = ModbusTcpClient(host=host, port=port)
    if not client.connect():
        raise RuntimeError(f"Failed to connect to Modbus TCP server {host}:{port}.")

    try:
        try:
            response = client.write_registers(start_address, list(values), slave=unit_id)
        except TypeError:
            response = client.write_registers(start_address, list(values), unit=unit_id)
        if response.isError():
            raise RuntimeError(f"Modbus write_registers failed: {response}")
    finally:
        client.close()
