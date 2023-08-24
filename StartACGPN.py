import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# 노트북 파일 경로 설정

def main():
    notebook_file = "main.ipynb"

    # 노트북 로드
    with open(notebook_file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 실행 프로세서 생성
    ep = ExecutePreprocessor(timeout=600)

    # 노트북 실행
    try:
        out = ep.preprocess(nb, {"metadata": {"path": "./"}})
    except Exception as e:
        out = None
        msg = f"Error executing the notebook: {str(e)}"

    # 실행 결과 출력
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        print(output['text'])

    # 실행 결과 메시지 출력
    if out is not None:
        print("Notebook execution completed successfully.")
    else:
        print("Notebook execution encountered an error:")
        print(msg)


