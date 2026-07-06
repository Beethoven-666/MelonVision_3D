from __future__ import annotations

from typing import Iterable


def _to_uint16_registers(values: Iterable[int]) -> list[int]:
    registers: list[int] = []
    for index, value in enumerate(values):
        int_value = int(value)
        if int_value < -32768 or int_value > 65535:
            raise ValueError(
                f"Register value at index {index} is out of 16-bit range: {int_value}. "
                "Use smaller PLC scaling or switch the PLC/Python mapping to 32-bit registers."
            )
        registers.append(int_value & 0xFFFF)
    return registers


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

    registers = _to_uint16_registers(values)
    client = ModbusTcpClient(host=host, port=port)
    if not client.connect():
        raise RuntimeError(f"Failed to connect to Modbus TCP server {host}:{port}.")

    try:
        try:
            response = client.write_registers(start_address, registers, slave=unit_id)
        except TypeError:
            response = client.write_registers(start_address, registers, unit=unit_id)
        if response.isError():
            raise RuntimeError(f"Modbus write_registers failed: {response}")
    finally:
        client.close()
