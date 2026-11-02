# GE 브라켓 스킨 보존(passive solid shell) 실험: zz, zzy 구현과 결과

작성: 2026-11-02. aieng/topopt 쪽 실험 기록이다. wht_topopt의 GE 브라켓 예제(`ex07_ge_bracket_advanced_algorithms.py`)에서 같은 시도를 재구현하고
개선할 수 있도록 방법, 수치, 한계를 정리했다. 코드는 서로 import하지 않으므로 아래 설명을 보고 wht 구조(`TopOptProblem.passive_solid`)에 맞게 다시 쓴다.

## 1. 왜 시도했나

GE 챌린지 우승작(M Arie Kurniawan, 327 g)은 안쪽에 삼각형 격벽을 가진 얇은 닫힌 껍질이다. 부피 제약 스윕과 응력 제약 최적화는 모두
트러스와 막대형 구조를 만들고 판(plate)형 구조를 만들지 못했다. 그래서 바깥 껍질 일부를 고정 솔리드로 강제하고 안쪽만 최적화하면
우승작 같은 껍질형이 나오는지 시험했다.

## 2. 구현 방법

### 2.1 스킨 요소 선택
설계 영역(`inside`, 격자에서 STL 영역 안에 있는 요소)의 노출면을 방향별로 구한다. 한 요소가 +x 면에 노출되었다는 것은 +x 이웃이
영역 밖이라는 뜻이다. 선택한 방향들의 노출 요소를 두께(요소 수)만큼 안쪽으로 키워서 `passive_solid`에 합집합으로 넣는다.

```python
def skin_elements(problem, inside, thickness_mm, faces):
    nx, ny, nz = problem.mesh.grid              # 요소 배열 순서는 (nz, ny, nx)
    h = float(problem.mesh.nodes[1, 0] - problem.mesh.nodes[0, 0])
    layers = max(1, int(round(thickness_mm / h)))
    ins = inside.reshape(nz, ny, nx)
    axis = {"x": 2, "y": 1, "z": 0}
    sel = np.zeros_like(ins)
    for f in faces:                              # 'x-' 'x+' 'y-' 'y+' 'z-' 'z+'
        ax, sgn = axis[f[0]], (1 if f[1] == "+" else -1)
        shifted = np.zeros_like(ins)             # 이웃 요소가 영역 안인가
        src, dst = [slice(None)] * 3, [slice(None)] * 3
        src[ax], dst[ax] = (slice(1, None), slice(0, -1)) if sgn > 0 else (slice(0, -1), slice(1, None))
        shifted[tuple(dst)] = ins[tuple(src)]
        layer = ins & ~shifted                   # 그 방향으로 노출된 요소
        grown = layer.copy()
        for _ in range(layers - 1):              # 두께만큼 안쪽으로 확장
            nxt = np.zeros_like(ins)
            s2, d2 = [slice(None)] * 3, [slice(None)] * 3
            s2[ax], d2[ax] = (slice(0, -1), slice(1, None)) if sgn > 0 else (slice(1, None), slice(0, -1))
            nxt[tuple(d2)] = grown[tuple(s2)]
            grown = grown | (nxt & ins)
        sel |= grown
    return np.flatnonzero(sel.ravel())

problem = replace(problem, passive_solid=np.union1d(problem.passive_solid, skin_elements(problem, inside, 2.0, ["z-", "z+"])))
```

### 2.2 사용한 조건
- 메시: MeshControl 88, 요소 2.0 mm, 격자 88x32x54, 설계 요소 44,682개, 자유도 약 168k (활성 기준). 해법은 PARDISO 직접법.
- 재료: Ti-6Al-4V, E 113.8 GPa, nu 0.342, 항복 903 MPa, 밀도 4.43 g/cm3. 4개 하중 케이스 (8000 lbf 위, 8500 lbf 바깥, 42도 9500 lbf, 토크 5000 lb-in), 강체 핀, 강체 볼트.
- 최적화: 질량 최소화, 하중 케이스마다 p-norm 응력 <= 903 MPa (qp 완화 q=0.5, p=8, 적응 스케일), MMA, 이동 한계 0.1, 시작 x=1.0.
  좌우 대칭(x), AM 오버행 필터(+y), 30회.
- 스킨 요소는 응력 제약에서 제외된다 (수동 솔리드). 하중점과 지지점 주변 수동 솔리드와 같은 취급이다.
- 최종 설계는 상위 밀도 요소를 같은 체적만큼 골라 0/1로 만들고 4개 하중으로 다시 풀어 요소 중심 von Mises를 확인한다.

### 2.3 스킨 면별 질량 (MC88, 2 mm, 1층)
| 면 | 요소 수 | 질량 |
|---|---|---|
| x- / x+ | 1,847 | 65 g 씩 |
| y- / y+ | 3,061 | 108 g 씩 |
| z- / z+ | 1,919 | 68 g 씩 |
| 6면 전체 | 10,284 | 364 g |

6면 전체 스킨만으로 364 g이다. 우승작 327 g보다 무겁다. 요소 2 mm 해상도에서는 닫힌 껍질 전체를 쓸 수 없다. 우승작 벽은 1.27 mm다.
참고: 2 mm 전체 껍질 질량 403 g (설계 영역 전체 표면 기준 추정), 1.27 mm 256 g, 3 mm 604 g.

## 3. 실험 결과 (MC88, 30회, 응력 903 MPa)

| 실험 | 스킨 면 | 스킨 요소 | 질량(0/1 설계) | 최대 von Mises 케이스 1/2/3/4 (MPa) | 99.5% (MPa) | 통과 (최대 / 99.5%) |
|---|---|---|---|---|---|---|
| zz | z-, z+ | 3,785 | 383 g | 976 / 2198 / 1117 / 545 | 692 / 1190 / 632 / 346 | 실패 / 실패 |
| zzy | z-, z+, y+ | 5,808 | 372 g | 1135 / 659 / 990 / 318 | 805 / 447 / 678 / 220 | 실패 / 통과 |
| 비교 A150 (스킨 없음, MC60, 150회) | - | - | 256 g | 920 / 772 / 752 / 504 | 813 / 589 / 601 / 429 | 실패 / 통과 |

주의: 비교 A150은 메시(MC60)와 반복(150회)이 달라 엄밀한 동일 조건 비교가 아니다. 스킨 실험은 30회라 응력 제약이 아직 풀리지 않은 상태에서 끝났다
(마지막 반복의 제약 g max zz +0.69, zzy +0.07).

해석:
- 스킨을 고정하면 질량이 늘고 응력이 개선되지 않았다. 스킨이 질량의 큰 부분을 차지해 안쪽에서 줄일 여지가 작다.
- zz는 케이스 2(바깥 8500 lbf)에서 2198 MPa까지 올라갔다. z 면 스킨만으로는 바깥 방향 하중 경로를 만들지 못했다.
- zzy의 y+ 스킨은 위쪽 판이 되어 모양이 쐐기형이다. 안쪽이 빈 격벽형 상자가 되지는 않았다.
- 우승작형 구조를 얻는 데는 단순한 고정 스킨이 아니라 얇은 벽을 표현할 수 있는 해상도(요소 1 mm 이하 또는 쉘 요소)와
  하중 경로에만 선택적으로 두는 스킨이 필요해 보인다.

## 4. 이전 실험과의 관계 (브리지)
좌우 두 반쪽을 잇는 바닥판(브리지 2.9 mm)을 수동 솔리드로 넣은 실험도 했다. 278 g, 최대 918/799/746/506 MPa로 스킨 없는 A150과 응력이 같고
질량만 22 g 늘었다. 강체 핀이 이미 좌우를 묶어서 중앙 연결은 하중 전달에 필요하지 않다는 해석과 일치한다.

## 5. wht_topopt에서 해볼 일
1. `ex07`에 `skin_elements` 같은 함수를 재구현하고 `passive_solid`에 합친다 (SE-AHCA, HYBRID_SE_AHCA, MMA에 같은 조건 적용).
2. 비교 지표는 질량, 면별 최대 von Mises, 99.5% von Mises, 케이스별 컴플라이언스를 모두 표로 남긴다.
3. 스킨 면 조합을 바꿔 본다. 이번에는 z±, z±y+만 했다. x±(하중 핀 쪽), y-(빌드 플레이트 면)는 시험하지 않았다.
4. 스킨 두께는 요소 수의 정수배다. MC88(2 mm)에서 1층만 시험했다. 해상도를 올려 1.27 mm 근처에서도 비교한다.
5. SE-AHCA에서는 스킨이 passive_solid 마스크로 들어가면 `hca.step`이 이를 xmax로 고정한다 (이미 지원).
6. 계산 시간: aieng 쪽은 PARDISO로 두 건을 병렬로 돌렸을 때 반복 한 번에 약 70초였다 (단독 실행 시간은 재지 않았다). 30회에 약 80분.

## 6. 재현 (aieng 쪽, 참고)
```
.venv/Scripts/python.exe aieng/examples/ge_bracket_challenge.py --mesh-control 88 --volfracs 0.35 --algos mma --am --symmetric \
  --solver pardiso --stress-limit 903 --iters 30 --skin 2 --skin-faces z- z+ [y+] -o <출력 폴더>
```
결과(PNG, STL, 요약표): `aieng/examples/output/ge_bracket_skin_zz/`, `aieng/examples/output/ge_bracket_skin_zzy/`.
