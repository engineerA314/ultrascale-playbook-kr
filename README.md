# UltraScale Playbook 한국어 번역 프로젝트

![image.png](images/title_image.png)

Ultra-Scale Playbook은 Hugging Face 의 nanotron 팀에서 작성한, 대규모 GPU 클러스터에서 LLM을 효율적으로 학습시키는 과정에 대한 종합 가이드입니다. 데이터·텐서·파이프라인·시퀀스/컨텍스트·전문가 병렬화와 ZeRO, 통신-계산 오버랩, 프로파일링/Nsight, 커널 최적화(예: FlashAttention), 그리고 실제 벤치마크와 코드 레퍼런스(picotron·nanotron)를 체계적으로, 그리고 구체적으로 다루는 글로, LLM 사전학습 (Pretraining) 에 대한 전반적이면서도 구체적인 지식을 얻을 수 있습니다.

이 저장소는 Hugging Face의 “The Ultra-Scale Playbook: Training LLMs on GPU Clusters”를 한국어로 번역한 비공식 프로젝트입니다.

번역 과정에서 용어 통일을 고려하며 번역하였으나, 미흡한 부분이 있을 수 있습니다.
원문에 존재하던 상호작용 가능하던 이미지 또는 Figure 들의 경우에는, 스크린샷 이미지로 대체하였습니다.

## 원문

- Space: https://huggingface.co/spaces/nanotron/ultrascale-playbook
- 인용(BibTeX):
  ```
  @misc{ultrascale_playbook,
        title={The Ultra-Scale Playbook: Training LLMs on GPU Clusters},
        author={Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf},
        year={2025},
  }
  ```

## 프로젝트 구성

- `The UltraScale Playbook.md`: The UltraScale Playbook 에 대한 한국어 번역본입니다.
- `NOTICE`, `LICENSE`: 라이선스 및 고지

## PDF 생성 (선택)

아래 도구 설치 후 명령으로 PDF를 생성할 수 있습니다.

- Pandoc 설치: `brew install pandoc`
- LaTeX(택1): `brew install --cask basictex` (권장) 또는 `brew install --cask mactex`
- 생성 명령:
  ```
  pandoc metadata.yaml "The UltraScale Playbook.md" \
    -o ultrascale-playbook-ko.pdf \
    --pdf-engine=xelatex \
    --top-level-division=chapter \
    --highlight-style=tango \
    -V colorlinks=true \
    -V linkcolor=blue \
    --resource-path=".:images" \
    --wrap=auto
  ```
  또는 스크립트 사용:
  ```
  bash scripts/build-pdf.sh
  ```

## Contribution

- 이슈/PR로 오타, 용어 통일, 추가 설명 제안은 언제나 환영합니다.
- 큰 편집은 이슈로 먼저 논의해 주세요.

## License

- 원문: Space 메타데이터에 `license: apache-2.0` 로 명시
- 본 번역본 및 부가 자료: Apache-2.0
- 세부 고지는 `NOTICE` 참조. 제3자 라이브러리는 각 파일의 라이선스를 따릅니다.

---

## (EN) Summary

- This repo hosts an unofficial Korean translation of “The Ultra-Scale Playbook: Training LLMs on GPU Clusters” (Hugging Face nanotron team).
- Files: `The UltraScale Playbook.md` (full text), `images/` (figures), `NOTICE`, `LICENSE`.
- License: Apache-2.0 (original and translation).
- PDF build: `bash scripts/build-pdf.sh` (uses `metadata.yaml`, XeLaTeX, `images/`).
