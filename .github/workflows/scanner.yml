name: Scanner UltraFast

on:
  workflow_dispatch:
  schedule:
    - cron: '*/5 * * * *'   # cada 5 minutos

jobs:
  run-scanner:
    runs-on: ubuntu-latest

    steps:
    - name: Clonar repositorio
      uses: actions/checkout@v3

    - name: Configurar Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Instalar dependencias
      run: |
        python -m pip install --upgrade pip
        pip install pandas numpy requests

    - name: Ejecutar scanner
      run: python ScannerUltraFast.py

    - name: Subir resultados como artifact
      uses: actions/upload-artifact@v4
      with:
        name: Scanner-UltraFast-Resultados
        path: backtest_escaneo.txt
