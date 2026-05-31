FROM public.ecr.aws/lambda/python:3.12

# OpenCV (headless) and easyocr still need a few shared libs at runtime.
RUN dnf install -y mesa-libGL libglvnd-glx && dnf clean all

# Install Python deps. torch CPU wheel is pulled from PyTorch's own index;
# everything else falls back to PyPI.
COPY requirements-lambda-container.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r ${LAMBDA_TASK_ROOT}/requirements-lambda-container.txt

# Pre-download EasyOCR models so cold start doesn't fetch from the network
# (Lambda has no outbound to S3 unless VPC/NAT is configured). Korean+English.
ENV EASYOCR_MODULE_PATH=/var/task/.EasyOCR
RUN python -c "import easyocr; easyocr.Reader(['ko','en'], gpu=False, model_storage_directory='/var/task/.EasyOCR/model', user_network_directory='/var/task/.EasyOCR/user_network')"

COPY ai_grading/ ${LAMBDA_TASK_ROOT}/ai_grading/
COPY pipeline.py ${LAMBDA_TASK_ROOT}/

CMD ["ai_grading.lambda_handler.handler"]
