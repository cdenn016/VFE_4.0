"""Frozen asymmetric Gaussian fixture for the sibling H2/H5 oracle."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np


Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]
VectorSeries: TypeAlias = tuple[Vector, ...]
MatrixSeries: TypeAlias = tuple[Matrix, ...]
RaggedVectorSeries: TypeAlias = tuple[Vector, ...]
RaggedMatrixSeries: TypeAlias = tuple[MatrixSeries, ...]

_SCHEMA_VERSION = "h2-h5-rectangular-v1"
_FIXTURE_ID = "peer-review-c5-pcg64-31337"
_GENERATOR = "numpy.random.PCG64"
_CANONICAL_DOMAIN = b"vfe4.h2-h5-rectangular-fixture.v1\x00"
_FROZEN_CANONICAL_B85 = """c-oa(O>-Q%vHUN*uDDV7Dge6r@UHK!*iaOJ!io_V$so0|pI`XDJ2BnEsh%O%whqhK>A_baD=V{l{{L=#{eJg3e8S&<+<o@=tB<ex^;ur@@yF}k=PzIAw@+W_liU9L$^Z7-r@Q&^_aA?}=+}pF_xbKw_2vFWe?I8n?t8j_?!R-q4=#j!_v!9;ebg8E@$`Z(KK}9j`Tu@>k;n1${n!8f%l~l17rs5cJpAM7@wA19$A=#e^4G8Z>8HFr$Ya;F{vG|@{VHxu7fD^~QJ#Ns7<s=wCm$$;A?&Dy=lk{f>FXT28_{)&#LvIb{&B9k<PdAly~J&n_3_A<v@`W2+}9*H>z_`aRjLG6R7&fYc~bSMarfH1ciXp5vy4J>v{Y4s%C?*PZJN-_%-t!K(==<{KiU2}^PfI0uJ$yfmd@?@7r%cTa~?QXDs|LucGGHXgL`Amk=EqrRo{uuN8A5k{`GN*%)O6{?I*hI#?6*{4&0jaV$XUL{Eve3(e^)>f8R}a(t1vrW2$R63%+NZeOCeSwrxZ&sglaTjn2zyx(m`I&8S-C`Tl&G5+b0HdCrmf`97RC5nV35_@KUtIg1NXMlE5|Ah%}0CcZxnwNO9h+L!&Fx#G|%#yN@wJ0)MHZ@#me=IFt*lOer&AI}&1X9K*1=|-rrCtojDltf#|U53U?+b+MnzrY9I-$Pa!x6oT=2^*?0&o)u$>f@NRhuPR1EH)n-pwxn@NwXNHhEk<;rF_QBpAAUTLl$M<+hP7nF0FqpM=j?Be|LqC2JCr1Jo%BO<e9^cru}GbPUTpA!{XlLc+fe0Xui$pcni`7<K~Ec<XM2_!==45;<VKXXtl_JMzh%oiik0l(7R84LGf*(n<YLPvBRiL<bL&4d#lI8O_l6=uhU7=3jpRI<l(x@*yClV)kjZ-2iDKsM*D;BPhw(ev#JlrTkABd8(mUtmZ)wP_-Me6icvjHW{o;%1LRx&Eebeg-kzv5J&qZ2p@rS^484xwB62b$j|KU>pgNpLLv!G&L&lWh%8;0S!O2YMV{fw&&xF4gC0fK!7YYj&f4+}z6Zz3Yb-;N?E^l|W9d)!?JLA4eOKZR$%E+l#o$5G68~Tm|xR@E!kS$G|3QiDfluX5Js0Z;O5s#duo?`zNvZpy;b(01g^0`HwskR>2P^4t$Q2Ow>2Q|Q_fCOR}Y8Z9cTMg`ZQgGFQ_6!WzQxyl$YLd1c_4GNocg$KR2ULzj)h<{O+Yi;dIn1FJ^k^18q!oMER7GabC5VC#@0~g9rh~rzO&@;$X3k9ZpFPCXN+SS()Tm#ALDX_x9GWGIy%*<$V3Wu^Y~b>J&oXmLpw=9Uq1Nn}1+_j`+VFPS=!^D;*N3M^Q%6Qz(I2A?U(C(;7Q(Cv`CoKiu;VGu(ODA^GA)?R_a|MxPc@Z@LZ4HpY@&%+0uG4^F5aC9>D>k18Q?E`RB;=v>1eZa+7EjUzaxnti{@vxXhvOez64OG%hTj3(iZ~KJ6}gYl+y(&c7$#tP9L)XbP?r^7I@G6ALa}+BXmZX<(#HBEAU$3c0e_u2D_IgZ8RG56r#7IjpRcWmONa_CP-@zp{6s>SS}0H3H3D_ZdxUtFYRVPHUoh5QetR`L^AUaDvljutv($X=w^kF2Al;~0l`U*$i2Mf(i=u!pZ<91@D?u*PcQ#LE-Bb5L0=Hb2qu?6&=8k0r_^k739*Oj%*c^cj3_HnN`yj&Qs^cnl^KWfxqw9dF_(V8qr5^QK0Us#&tPqobEuj|l1)FILgJx2T+Lk_<!mxyHW0YPks76P&@u0f*c5%41D2zf5scu<6AarOhuXF>{5H?c3LgwOt5;4a9(h2I^0CrW@g@3HVS9)#6LH5hP)Zo0pa6HCi3e5Yh$<IA=H@Ow8sN3m?kAIviELpEQI}Fr0KOh3x?bU<0lSrJ#mCB_Ek?OIT7t6A4x~~wn_H_*YAuN;kXB5I1-gsRB95W-54ZNth)tn62jp>|<cuqvF?CYZQIs7%r0C5G9}L*1ClBj0Gj7=Pm5ezd_W9{$D^dG_IfEc!Ur2QvBxyn%#I*=}X)-tGHj6a)xn9FTNTOy9n81>?cHBX|7{eH{AMkE4LL0^GgdT{XhFyb(J;-fcCTJ=Y^e`qXyDXp!%pa#!OI}<W5TlVM?G<3%=FSe)XP+F8%GM({Wdfy(@XTW93ZUz?*5=%j&j*hmfuw+X3n6E2!6P?%2r&cz%52IZ4jBe7p}5Lnq6EGwz&CVE=gaA4fF@YRgy7LFqf_uH^~^)cN!M&p-KNUD^^Ti!K8dM6j^XsR`;wirjRX!j0gSgtM%dClD9027=h|X4^CVoF4%{bH^aWuc8<@5xK4Gi&-T;6KdkjK`KQM@NC;<_COk){|Gms=ws*oHRbtA2sdcL;Jl|WN6XofhsW)gbBjet`fw>g6C19XVfS`a-rlLMG$$_Zgzq-UwbX=yRX3Ar_GODMtW!Fsah*$-)0xd=t*@6{zZOoZB(ogdJ_)wpMkBMMi@v$9vXw|k34YO2tqO`;esksMsF5E~%;WHd5@3W1M7RNaBl8}zsX6E{dSHPIjmnqnT=do{ENgcaO>M}q1+3!KxH>QLaXg_y)F41)s;*v0~KHocpL;Fv(UvZ>b#dm?&pT?%^(O=?igpmH8XIu<|Zl|=$50n^3N4@Z+ZYD|o(jHIycE%iZyEYwsDRuP4f8-oN=85Y}w5(6ytA|sJ)5GC)B3NyD}k~y0zA-feO9B&sU;LoS_sxLp~<6`d<dzX@fJ!yw#j86lG5>95Oa=_EglbxBM7M3|fX>ap_o73JTDhi!QMHDYcqr_Z5mU;TttunfKu*1ngu#Pggj&jv`ff$R(iiR|9bRoHbjN?KRNkq{G_vXQl{nUG!-O(zsTY0^T;duD2kIRRgn#uObZn))n5AJs9kYUifQT=GPJ(5X-cfARaZJEwbRS842Aw8_Q2ta<R$d1*KG*U;AlHOMs(OhjFg{0IX+%Jm4RT4{K&r}P9EX_-KC&&hvM4dHc(>4iE90Pg@f$B$~dno*-u!7LUYQ6hxHManSRvG5ZVGQuKba;k9R|TOBr;sn*gPDy_ggYq=>(pU6$#5N{gP%}+P?7UkqY5Rc%`ge&(^e3I7Diohnb=xmpI2oZkW*!AWZYc1Z3k?b4?EjuMG1y}HJ#Me$w0kPr~*3X3up(K0&>L8Y^R{10#;aZDoGESAJ**z=a|S}R`4BADwKUgw(jshZ1p{ljBTM5K@$QcMr=WkuioDn%j8;|*$2s>AhEdIV{!=O-Y+@7x(HllHZG9W2MF@&6{EhmI`tS3^Td?r(zLr6AsD5(BRlu*O2XN}RNY!@&44nbq%O~QD#hnDZl3J7gj|sn#Ru1PbEa28eElIm^g4|Psh3z4HDH*ge1Q<FrKaJW9P8Waw5uE1p?{pItz4%={RelJt1yn>2Z|`fee#E1b`ghiMJNaJ*tB+Vr$7R7-}d5RJ~9+=h-0*VmHA;}P%pj1Uiu45qM+BwzK0h1(1@hwArS(B*Tu3_3bQAI^MQVC?`6%!{jLsA@HHUCvXw*7yt<fC38LA`9_nSHj6kcZ*bMH?$ACfw!79}TqIQ8ZSzY%Ut(0&;5-b$1f63Lo#rN93>38}1Q(ra?zx(FzzomO5&JTI~{g1!O%l*$a!z(<+-RIL=KK#!K67O4OXk`chS>?^DXat(TQ4NB8<Hn1Xy?(MYT?TQ)49xa0j>v!x^nSJiEg&|8Jv*f7BoJ6}hXlKMa%T9f@G}#pBE)0;M7juP*fj4n9@c0rts*s<kKOm1C%aOcxY-PU@1X^+&cQIc6#H1<F?Mit@}>H^S`PNj+0BD<-we`&U?4)_OO4*s_O0E$iGt8V%>&Z7+KwwMfJpW#NQ8DMAKbAVY>QVq*!PT*)=|`24|*ICd`{KRIoC@;3s=kyNK6Xgj;nXB4k8u;KClMs;9`)`AczZS<qfafvEGvwX;po=KG21l-ilK$(fQSXh9Wj<zn7Z9F`hJDZHP)BIPbyX@M<R}o$y!9=UI$kg?7*xMhbDgEqkNsz0BsJ07WWXs*wa`*5qY!2yj3buy`R>07VKe`Pki)JbHC0LIh8P$C73N5kQTvVsm+ckyhgaK6i%67iqD*N-Im$DTG0=<q2wI<N}@9r*~$n9Ujz2oTErOwBM)(X2IEGU(KdB{qD0k4_HImSeed!!1#cfCYSQoDBSMz`V~V>4jEHd1@QK?_^3^o`WlLFhwRI_cj$rj4l>U}(WT~U1@=zABQO;S+sUwPoX5$%UtZKjMzz-nT_pWvt=_iz-@Y7Ra-%SGHDs-I^|Qwb;H{TH2ZsYaMs>A2{m90=;ksN;j%8(#c{$*61N%cGw4C;)@_D202%o5Fz}GV+Cv3X<v|e_V)&6#JYTUiu@6D6*>otKfBv&b9I{KPeC)`MaN_DHn@wQVSg(h5mp7qdcwh!WEPq;4BYj_(6#2;4(d0i$7CGx<cH~J9LC`8DTd_abMV!Fhm<QPySApzE_m0;ocU-qqwLSn-@l-|_zfjUwt!#ly{^1-^aK~RSSX$w(ls*x({e|91?)mZoFXoB3A{B7sH^|sG{*S9VZh<@1~O&Rs=(&+?2u3vF@$WA(_#G5C3!wxDQ!nx&1%vWmHMc6j5ZdgB<xTJg6mgaifuA?2>FE>y2=F*r1EQbm+QLYN5N~2bmy=L1r@6oT)3iX`4y^xy+`}e6sgEo>%C_%?U`EEmXNbMfr_mt1Q=+ph>(wrs+r<WtAM2*cgN6sY)hK)1IXpU)_Fx}YLG{}7PWy%~RtoDVAo5e|WFyEma6k*LiyQ~A7tCqQ-;1~OyUh9{uZjKgcZ+5k%cJK?RCgStJDk8^%<8kh$@Hg?$BwGayb*>EPwtAMjZ)ZXBrCf1UP)=$_wZfsb2{#ijXW34l*3We$x+uX4Mjm;<N1&P`_*angWe5}xDo>2I>DH+!v?qsgkuj&aX7do4>qjbVez^pYR2|FcQoLRwpc3QmQNOzpj9z#m8QnyXgGC&?WJ`(RWLDqxqCMb)P(2wN>vboY-~Rb8bpCsx"""

# Filled from the exact PCG64 construction below and checked on every load.
H2_H5_RECTANGULAR_RAW_SHA256 = (
    "6925ffe08e4d8acbc7790b6318f3e26a0509a8208ebf062f62f721332d194aa5"
)
H2_H5_RECTANGULAR_CANONICAL_SHA256 = (
    "02add1038f70cedd2cb5b0adad0c3b23696960f9fe2a0c4942df7eea77e3f58c"
)


@dataclass(frozen=True, slots=True)
class H2H5RectangularFixture:
    """Owned immutable arrays reproducing the peer-review C5 Gaussian case."""

    schema_version: Literal["h2-h5-rectangular-v1"]
    fixture_id: Literal["peer-review-c5-pcg64-31337"]
    generator: Literal["numpy.random.PCG64"]
    seed: int
    horizon: int
    d_z: int
    d_m: int
    observation_dimension: int
    dense_parents: tuple[tuple[int, ...], ...]
    state_source_priors: RaggedVectorSeries
    model_source_priors: RaggedVectorSeries
    state_parent_weights: RaggedVectorSeries
    model_parent_weights: RaggedVectorSeries
    state_transports: RaggedMatrixSeries
    model_transports: RaggedMatrixSeries
    state_model_maps: MatrixSeries
    state_offsets: VectorSeries
    model_offsets: VectorSeries
    state_transition_covariances: MatrixSeries
    model_transition_covariances: MatrixSeries
    state_precisions: MatrixSeries
    model_precisions: MatrixSeries
    initial_mean: Vector
    initial_covariance: Matrix
    observation_state_maps: MatrixSeries
    observation_model_maps: MatrixSeries
    observation_offsets: VectorSeries
    observation_covariances: MatrixSeries
    observation_precisions: MatrixSeries
    observations: VectorSeries
    state_means: VectorSeries
    state_covariances: MatrixSeries
    model_means: VectorSeries
    model_covariances: MatrixSeries
    raw_sha256: str = field(init=False)
    canonical_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.schema_version,
            self.fixture_id,
            self.generator,
            self.seed,
            self.horizon,
            self.d_z,
            self.d_m,
            self.observation_dimension,
        ) != (
            _SCHEMA_VERSION,
            _FIXTURE_ID,
            _GENERATOR,
            31337,
            3,
            2,
            3,
            2,
        ):
            raise ValueError("rectangular fixture identity does not match C5")
        expected_parents = tuple(tuple(range(time)) for time in range(1, 4))
        if self.dense_parents != expected_parents:
            raise ValueError("dense_parents must equal range(t) for t=1,2,3")
        _validate_fixture_shapes(self)
        object.__setattr__(
            self,
            "raw_sha256",
            hashlib.sha256(h2_h5_rectangular_raw_bytes(self)).hexdigest(),
        )
        object.__setattr__(
            self,
            "canonical_sha256",
            hashlib.sha256(
                _CANONICAL_DOMAIN + h2_h5_rectangular_canonical_bytes(self)
            ).hexdigest(),
        )


def _simplex(rng: np.random.Generator, size: int) -> np.ndarray:
    values = rng.random(size) + 0.25
    return values / values.sum()


def _determinant_small(value: np.ndarray) -> float:
    """Return a 2x2/3x3 determinant without dispatching to BLAS/LAPACK."""

    rows = tuple(
        tuple(float(item) for item in row)
        for row in np.asarray(value, dtype=np.float64)
    )
    if len(rows) == 2 and all(len(row) == 2 for row in rows):
        return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    if len(rows) == 3 and all(len(row) == 3 for row in rows):
        return math.fsum(
            (
                rows[0][0]
                * (
                    rows[1][1] * rows[2][2]
                    - rows[1][2] * rows[2][1]
                ),
                -rows[0][1]
                * (
                    rows[1][0] * rows[2][2]
                    - rows[1][2] * rows[2][0]
                ),
                rows[0][2]
                * (
                    rows[1][0] * rows[2][1]
                    - rows[1][1] * rows[2][0]
                ),
            )
        )
    raise ValueError("determinant replay supports only 2x2 and 3x3 matrices")


def _invertible(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=(dimension, dimension))
    while abs(_determinant_small(value)) < 0.3:
        value = rng.normal(size=(dimension, dimension))
    return value


def _spd(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=(dimension, dimension))
    result = np.empty((dimension, dimension), dtype=np.float64)
    for row in range(dimension):
        for column in range(dimension):
            result[row, column] = math.fsum(
                float(value[row, index]) * float(value[column, index])
                for index in range(dimension)
            ) + (0.8 if row == column else 0.0)
    return result


def _inverse_small(value: Matrix) -> Matrix:
    """Invert the frozen fixture's tiny matrices with scalar Gauss-Jordan."""

    dimension = len(value)
    if dimension < 1 or any(len(row) != dimension for row in value):
        raise ValueError("small inverse requires a nonempty square matrix")
    augmented = [
        [float(item) for item in row]
        + [1.0 if row_index == column else 0.0 for column in range(dimension)]
        for row_index, row in enumerate(value)
    ]
    for column in range(dimension):
        pivot_row = max(
            range(column, dimension),
            key=lambda row: abs(augmented[row][column]),
        )
        pivot = augmented[pivot_row][column]
        if pivot == 0.0 or not math.isfinite(pivot):
            raise ValueError("small inverse requires a nonsingular matrix")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )
        pivot = augmented[column][column]
        augmented[column] = [item / pivot for item in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                left - multiplier * right
                for left, right in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    inverse = [
        [row[dimension + column] for column in range(dimension)]
        for row in augmented
    ]
    return tuple(
        tuple(
            0.5 * (inverse[row][column] + inverse[column][row])
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _vector(value: np.ndarray) -> Vector:
    array = np.asarray(value, dtype=np.float64)
    return tuple(float(item) for item in array)


def _matrix(value: np.ndarray) -> Matrix:
    array = np.asarray(value, dtype=np.float64)
    return tuple(tuple(float(item) for item in row) for row in array)


def _build_fixture_from_seed() -> H2H5RectangularFixture:
    rng = np.random.Generator(np.random.PCG64(31337))
    horizon, d_z, d_m, observation_dimension = 3, 2, 3, 2
    dense_parents = tuple(tuple(range(time)) for time in range(1, horizon + 1))

    state_source_priors = tuple(
        _vector(_simplex(rng, len(parents))) for parents in dense_parents
    )
    model_source_priors = tuple(
        _vector(_simplex(rng, len(parents))) for parents in dense_parents
    )
    state_parent_weights = tuple(
        _vector(_simplex(rng, len(parents))) for parents in dense_parents
    )
    model_parent_weights = tuple(
        _vector(_simplex(rng, len(parents))) for parents in dense_parents
    )
    state_transports = tuple(
        tuple(_matrix(_invertible(rng, d_z)) for _ in parents)
        for parents in dense_parents
    )
    model_transports = tuple(
        tuple(_matrix(_invertible(rng, d_m)) for _ in parents)
        for parents in dense_parents
    )
    state_model_maps = tuple(
        _matrix(rng.normal(size=(d_z, d_m))) for _ in range(horizon)
    )
    state_offsets = tuple(_vector(rng.normal(size=d_z)) for _ in range(horizon))
    model_offsets = tuple(_vector(rng.normal(size=d_m)) for _ in range(horizon))
    state_transition_covariances = tuple(
        _matrix(_spd(rng, d_z)) for _ in range(horizon)
    )
    model_transition_covariances = tuple(
        _matrix(_spd(rng, d_m)) for _ in range(horizon)
    )
    state_precisions = tuple(
        _inverse_small(value)
        for value in state_transition_covariances
    )
    model_precisions = tuple(
        _inverse_small(value)
        for value in model_transition_covariances
    )

    initial_mean = _vector(rng.normal(size=d_z + d_m))
    initial_covariance = _matrix(_spd(rng, d_z + d_m))
    observation_state_maps = tuple(
        _matrix(rng.normal(size=(observation_dimension, d_z)))
        for _ in range(horizon)
    )
    observation_model_maps = tuple(
        _matrix(rng.normal(size=(observation_dimension, d_m)))
        for _ in range(horizon)
    )
    observation_offsets = tuple(
        _vector(rng.normal(size=observation_dimension)) for _ in range(horizon)
    )
    observation_covariances = tuple(
        _matrix(_spd(rng, observation_dimension)) for _ in range(horizon)
    )
    observation_precisions = tuple(
        _inverse_small(value)
        for value in observation_covariances
    )
    observations = tuple(
        _vector(rng.normal(size=observation_dimension)) for _ in range(horizon)
    )

    # The peer-review constructor draws the unused categorical-emission arrays
    # before the Gaussian recognition arrays. Consume those draws exactly.
    vocabulary_size = 4
    for _ in range(horizon):
        rng.normal(size=(vocabulary_size, d_z))
    for _ in range(horizon):
        rng.normal(size=(vocabulary_size, d_m))
    for _ in range(horizon):
        rng.normal(size=vocabulary_size)
    for _ in range(horizon):
        rng.integers(vocabulary_size)

    state_means = tuple(
        _vector(rng.normal(size=d_z)) for _ in range(horizon + 1)
    )
    state_covariances = tuple(
        _matrix(_spd(rng, d_z)) for _ in range(horizon + 1)
    )
    model_means = tuple(
        _vector(rng.normal(size=d_m)) for _ in range(horizon + 1)
    )
    model_covariances = tuple(
        _matrix(_spd(rng, d_m)) for _ in range(horizon + 1)
    )

    return H2H5RectangularFixture(
        schema_version=_SCHEMA_VERSION,
        fixture_id=_FIXTURE_ID,
        generator=_GENERATOR,
        seed=31337,
        horizon=horizon,
        d_z=d_z,
        d_m=d_m,
        observation_dimension=observation_dimension,
        dense_parents=dense_parents,
        state_source_priors=state_source_priors,
        model_source_priors=model_source_priors,
        state_parent_weights=state_parent_weights,
        model_parent_weights=model_parent_weights,
        state_transports=state_transports,
        model_transports=model_transports,
        state_model_maps=state_model_maps,
        state_offsets=state_offsets,
        model_offsets=model_offsets,
        state_transition_covariances=state_transition_covariances,
        model_transition_covariances=model_transition_covariances,
        state_precisions=state_precisions,
        model_precisions=model_precisions,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        observation_state_maps=observation_state_maps,
        observation_model_maps=observation_model_maps,
        observation_offsets=observation_offsets,
        observation_covariances=observation_covariances,
        observation_precisions=observation_precisions,
        observations=observations,
        state_means=state_means,
        state_covariances=state_covariances,
        model_means=model_means,
        model_covariances=model_covariances,
    )


def _decode_frozen_value(value: object) -> object:
    if type(value) is str and (
        value.startswith("0x") or value.startswith("-0x")
    ):
        return float.fromhex(value)
    if type(value) is list:
        return tuple(_decode_frozen_value(item) for item in value)
    if type(value) in (str, int):
        return value
    raise ValueError("frozen rectangular fixture payload is malformed")


def _build_fixture() -> H2H5RectangularFixture:
    try:
        canonical_bytes = zlib.decompress(
            base64.b85decode(_FROZEN_CANONICAL_B85.encode("ascii"))
        )
        raw = json.loads(canonical_bytes.decode("ascii"))
    except (UnicodeError, ValueError, zlib.error) as exc:
        raise RuntimeError("frozen rectangular fixture could not be decoded") from exc
    if type(raw) is not dict:
        raise RuntimeError("frozen rectangular fixture root must be an object")
    decoded = {
        name: _decode_frozen_value(value) for name, value in raw.items()
    }
    return H2H5RectangularFixture(**decoded)  # type: ignore[arg-type]


def _validate_fixture_shapes(fixture: H2H5RectangularFixture) -> None:
    t, d_z, d_m, d_x = (
        fixture.horizon,
        fixture.d_z,
        fixture.d_m,
        fixture.observation_dimension,
    )
    for name in (
        "state_source_priors",
        "model_source_priors",
        "state_parent_weights",
        "model_parent_weights",
    ):
        rows = getattr(fixture, name)
        if len(rows) != t:
            raise ValueError(f"{name} must contain one row per transition")
        for time, row in enumerate(rows, start=1):
            _require_array(row, (time,), name)
            if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=2.0e-15):
                raise ValueError(f"{name}[{time}] must sum to one")
            if any(value <= 0.0 for value in row):
                raise ValueError(f"{name}[{time}] must be strictly positive")
    for name, dimension in (
        ("state_transports", d_z),
        ("model_transports", d_m),
    ):
        rows = getattr(fixture, name)
        if len(rows) != t:
            raise ValueError(f"{name} must contain one row per transition")
        for time, row in enumerate(rows, start=1):
            if len(row) != time:
                raise ValueError(f"{name}[{time}] must contain range(t)")
            for matrix in row:
                _require_array(matrix, (dimension, dimension), name)
    _require_series(fixture.state_model_maps, t, (d_z, d_m), "state_model_maps")
    _require_series(fixture.state_offsets, t, (d_z,), "state_offsets")
    _require_series(fixture.model_offsets, t, (d_m,), "model_offsets")
    for name, values, dimension in (
        (
            "state_transition_covariances",
            fixture.state_transition_covariances,
            d_z,
        ),
        (
            "model_transition_covariances",
            fixture.model_transition_covariances,
            d_m,
        ),
        ("state_precisions", fixture.state_precisions, d_z),
        ("model_precisions", fixture.model_precisions, d_m),
    ):
        _require_series(values, t, (dimension, dimension), name)
        for value in values:
            _require_spd(value, name)
    _require_inverse_series(
        fixture.state_transition_covariances,
        fixture.state_precisions,
        "state transition covariance/precision",
    )
    _require_inverse_series(
        fixture.model_transition_covariances,
        fixture.model_precisions,
        "model transition covariance/precision",
    )
    _require_array(fixture.initial_mean, (d_z + d_m,), "initial_mean")
    _require_array(
        fixture.initial_covariance,
        (d_z + d_m, d_z + d_m),
        "initial_covariance",
    )
    _require_spd(fixture.initial_covariance, "initial_covariance")
    _require_series(
        fixture.observation_state_maps,
        t,
        (d_x, d_z),
        "observation_state_maps",
    )
    _require_series(
        fixture.observation_model_maps,
        t,
        (d_x, d_m),
        "observation_model_maps",
    )
    _require_series(
        fixture.observation_offsets, t, (d_x,), "observation_offsets"
    )
    for name, values in (
        ("observation_covariances", fixture.observation_covariances),
        ("observation_precisions", fixture.observation_precisions),
    ):
        _require_series(values, t, (d_x, d_x), name)
        for value in values:
            _require_spd(value, name)
    _require_inverse_series(
        fixture.observation_covariances,
        fixture.observation_precisions,
        "observation covariance/precision",
    )
    _require_series(fixture.observations, t, (d_x,), "observations")
    _require_series(fixture.state_means, t + 1, (d_z,), "state_means")
    _require_series(
        fixture.state_covariances, t + 1, (d_z, d_z), "state_covariances"
    )
    _require_series(fixture.model_means, t + 1, (d_m,), "model_means")
    _require_series(
        fixture.model_covariances, t + 1, (d_m, d_m), "model_covariances"
    )
    for name, values in (
        ("state_covariances", fixture.state_covariances),
        ("model_covariances", fixture.model_covariances),
    ):
        for value in values:
            _require_spd(value, name)


def _require_series(
    values: tuple[object, ...],
    length: int,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if type(values) is not tuple or len(values) != length:
        raise ValueError(f"{name} must contain {length} arrays")
    for value in values:
        _require_array(value, shape, name)


def _require_array(value: object, shape: tuple[int, ...], name: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{name} must use immutable tuples")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must have finite shape {shape}")


def _require_spd(value: Matrix, name: str) -> None:
    array = np.asarray(value, dtype=np.float64)
    if not np.allclose(array, array.T, rtol=0.0, atol=1.0e-14):
        raise ValueError(f"{name} must be symmetric to binary64 solve precision")
    symmetric = 0.5 * (array + array.T)
    lower = [[0.0] * len(array) for _ in range(len(array))]
    for row in range(len(array)):
        for column in range(row + 1):
            residual = float(symmetric[row, column]) - math.fsum(
                lower[row][index] * lower[column][index]
                for index in range(column)
            )
            if row == column:
                if residual <= 0.0 or not math.isfinite(residual):
                    raise ValueError(f"{name} must be positive definite")
                lower[row][column] = math.sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]


def _require_inverse_series(
    covariances: MatrixSeries,
    precisions: MatrixSeries,
    name: str,
) -> None:
    if len(covariances) != len(precisions):
        raise ValueError(f"{name} series lengths must match")
    for time, (covariance, precision) in enumerate(
        zip(covariances, precisions, strict=True)
    ):
        dimension = len(covariance)
        for row in range(dimension):
            for column in range(dimension):
                product = math.fsum(
                    covariance[row][index] * precision[index][column]
                    for index in range(dimension)
                )
                expected = 1.0 if row == column else 0.0
                if not math.isclose(
                    product,
                    expected,
                    rel_tol=0.0,
                    abs_tol=2.0e-13,
                ):
                    raise ValueError(
                        f"{name}[{time}] does not multiply to identity"
                    )


def _array_records(
    fixture: H2H5RectangularFixture,
) -> tuple[tuple[str, np.ndarray], ...]:
    records: list[tuple[str, np.ndarray]] = []
    for name in (
        "state_source_priors",
        "model_source_priors",
        "state_parent_weights",
        "model_parent_weights",
        "state_model_maps",
        "state_offsets",
        "model_offsets",
        "state_transition_covariances",
        "model_transition_covariances",
        "state_precisions",
        "model_precisions",
        "initial_mean",
        "initial_covariance",
        "observation_state_maps",
        "observation_model_maps",
        "observation_offsets",
        "observation_covariances",
        "observation_precisions",
        "observations",
        "state_means",
        "state_covariances",
        "model_means",
        "model_covariances",
    ):
        value = getattr(fixture, name)
        if name in {
            "state_source_priors",
            "model_source_priors",
            "state_parent_weights",
            "model_parent_weights",
        }:
            for index, item in enumerate(value):
                records.append((f"{name}[{index}]", np.asarray(item)))
        else:
            records.append((name, np.asarray(value)))
    for name in ("state_transports", "model_transports"):
        for time, row in enumerate(getattr(fixture, name), start=1):
            for parent, item in enumerate(row):
                records.append((f"{name}[{time},{parent}]", np.asarray(item)))
    return tuple(records)


def h2_h5_rectangular_raw_bytes(fixture: H2H5RectangularFixture) -> bytes:
    """Return the exact little-endian binary fixture payload."""

    if type(fixture) is not H2H5RectangularFixture:
        raise ValueError("fixture must be H2H5RectangularFixture")
    metadata = json.dumps(
        {
            "schema_version": fixture.schema_version,
            "fixture_id": fixture.fixture_id,
            "generator": fixture.generator,
            "seed": fixture.seed,
            "horizon": fixture.horizon,
            "d_z": fixture.d_z,
            "d_m": fixture.d_m,
            "observation_dimension": fixture.observation_dimension,
            "dense_parents": fixture.dense_parents,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    chunks = [struct.pack("<Q", len(metadata)), metadata]
    for name, value in _array_records(fixture):
        label = name.encode("ascii")
        array = np.ascontiguousarray(value, dtype="<f8")
        chunks.extend(
            (
                struct.pack("<Q", len(label)),
                label,
                struct.pack("<Q", array.ndim),
                struct.pack(f"<{array.ndim}Q", *array.shape),
                struct.pack("<Q", array.nbytes),
                array.tobytes(order="C"),
            )
        )
    return b"".join(chunks)


def _canonicalize(value: object) -> object:
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical fixture floats must be finite")
        return value.hex()
    if type(value) in (str, int):
        return value
    if type(value) is tuple:
        return [_canonicalize(item) for item in value]
    raise ValueError(f"unsupported canonical fixture value {type(value).__name__}")


def _canonical_core(fixture: H2H5RectangularFixture) -> dict[str, object]:
    return {
        name: _canonicalize(getattr(fixture, name))
        for name in (
            "schema_version",
            "fixture_id",
            "generator",
            "seed",
            "horizon",
            "d_z",
            "d_m",
            "observation_dimension",
            "dense_parents",
            "state_source_priors",
            "model_source_priors",
            "state_parent_weights",
            "model_parent_weights",
            "state_transports",
            "model_transports",
            "state_model_maps",
            "state_offsets",
            "model_offsets",
            "state_transition_covariances",
            "model_transition_covariances",
            "state_precisions",
            "model_precisions",
            "initial_mean",
            "initial_covariance",
            "observation_state_maps",
            "observation_model_maps",
            "observation_offsets",
            "observation_covariances",
            "observation_precisions",
            "observations",
            "state_means",
            "state_covariances",
            "model_means",
            "model_covariances",
        )
    }


def h2_h5_rectangular_canonical_bytes(
    fixture: H2H5RectangularFixture,
) -> bytes:
    """Return the canonical exact-hex JSON preimage (without its domain)."""

    if type(fixture) is not H2H5RectangularFixture:
        raise ValueError("fixture must be H2H5RectangularFixture")
    return json.dumps(
        _canonical_core(fixture),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _require_seed_replay(fixture: H2H5RectangularFixture) -> None:
    """Bind the frozen payload to the declared PCG64 seed and draw order."""

    replay = _build_fixture_from_seed()
    for name in (
        "schema_version",
        "fixture_id",
        "generator",
        "seed",
        "horizon",
        "d_z",
        "d_m",
        "observation_dimension",
        "dense_parents",
    ):
        if getattr(replay, name) != getattr(fixture, name):
            raise RuntimeError(f"rectangular fixture seed replay drifted at {name}")
    fixture_records = dict(_array_records(fixture))
    for name, replay_array in _array_records(replay):
        frozen_array = fixture_records[name]
        if replay_array.shape != frozen_array.shape:
            raise RuntimeError(
                f"rectangular fixture seed replay shape drifted at {name}"
            )
        maximum_error = max(
            (
                abs(float(left) - float(right))
                for left, right in zip(
                    replay_array.flat, frozen_array.flat, strict=True
                )
            ),
            default=0.0,
        )
        if maximum_error > 5.0e-13:
            raise RuntimeError(
                f"rectangular fixture PCG64 seed replay drifted at {name}"
            )


def load_h2_h5_rectangular_fixture() -> H2H5RectangularFixture:
    """Reproduce and hash-check the immutable PCG64 C5 fixture."""

    fixture = _build_fixture()
    if fixture.raw_sha256 != H2_H5_RECTANGULAR_RAW_SHA256:
        raise RuntimeError("rectangular fixture raw SHA-256 drifted")
    if fixture.canonical_sha256 != H2_H5_RECTANGULAR_CANONICAL_SHA256:
        raise RuntimeError("rectangular fixture canonical SHA-256 drifted")
    _require_seed_replay(fixture)
    return fixture


__all__ = [
    "H2H5RectangularFixture",
    "H2_H5_RECTANGULAR_CANONICAL_SHA256",
    "H2_H5_RECTANGULAR_RAW_SHA256",
    "h2_h5_rectangular_canonical_bytes",
    "h2_h5_rectangular_raw_bytes",
    "load_h2_h5_rectangular_fixture",
]
