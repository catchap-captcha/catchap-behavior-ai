# 지영님께 — 프록시 소유 주체 회신 (2026-08-05)

조성원 · 실측 근거 포함

---

## 물어보신 (a)/(b) — 둘 다 아닙니다

프록시는 **프론트 엣지의 Caddy** 입니다. 민서 쪽 서버가 아닙니다.

```
dig www.catchap5.com        →  210.109.14.25   (프론트 VM)
dig captcha.catchap5.com    →  응답 없음        (아직 미등록)

curl -sD- https://www.catchap5.com/
  HTTP/2 200   server: nginx/1.27.5   via: 1.1 Caddy

curl -sD- https://www.catchap5.com/captcha-api/api/config
  HTTP/2 200   server: uvicorn        via: 1.1 Caddy
```

**둘 다 `via: 1.1 Caddy` 입니다.** 같은 엣지가 `/` 는 nginx 로, `/captcha-api` 는
캡차 uvicorn 으로 갈라 보내고 있습니다.

지영님이 nginx 설정에서 못 찾으신 게 맞습니다 — **nginx 보다 앞단**에 있습니다.
민서 서버(61.109.239.231)에 있는 게 아니라, 프론트 IP 로 들어온 요청을 그 Caddy 가
갈라주는 구조입니다.

그래서 `captcha.catchap5.com` 도 **같은 Caddy 에 사이트 블록 하나 추가**하는 일입니다.
그 Caddy 를 지영님이 만지실 수 있으면 지영님 쪽이고, 인프라 담당이 세운 것이면
그쪽으로 넘기시면 됩니다. **민서 쪽 일은 아닙니다.**

---

## 필요한 것은 두 개입니다

### ① DNS — 가비아

네임서버를 확인했더니 가비아입니다.

```
NS   ns.gabia.net.  ns1.gabia.co.kr.  ns.gabia.co.kr.
```

```
타입      A
호스트    captcha              ← 이것만 (.catchap5.com 은 자동)
값        210.109.14.25        ← 기존 www 와 같은 IP
TTL       600
```

`catchap5.com` 과 `www.catchap5.com` 이 이미 `210.109.14.25` 라 일관됩니다.

**가비아 계정을 누가 갖고 계신지 알려주세요.** 제 접근이 되면 제가 넣겠습니다.

### ② Caddy 사이트 블록

```
captcha.catchap5.com {
    reverse_proxy 61.109.239.231:8000
}
```

**루트 그대로여야 합니다.** `handle_path` 로 경로를 잘라 붙이면 어제 말씀드린
세 군데(widget.js origin · `/assets` · `/api`)가 그대로 다시 깨집니다.

---

## 순서 — DNS 가 먼저입니다

인증서를 확인했는데 와일드카드가 아닙니다.

```
subject  CN=www.catchap5.com
SAN      DNS:www.catchap5.com
```

`captcha.catchap5.com` 은 인증서를 새로 받아야 합니다. Caddy 가 자동으로 받아주지만
**DNS 가 먼저 응답해야** ACME 검증이 통과합니다.

그리고 SOA 의 negative caching TTL 이 깁니다.

```
SOA  ... 1800 600 1209600 86400
                            ↑ "없는 이름" 캐시 수명 = 24시간
```

**"이 이름은 없다"는 응답도 캐시됩니다.** 저와 지영님이 이미 조회해버려서 일부
리졸버에 박혀 있을 수 있습니다. A 레코드 넣자마자 Caddy 를 켜면 발급이 실패하고
재시도 대기에 들어갑니다.

```
1  가비아에 A 레코드 등록
2  dig @ns.gabia.co.kr captcha.catchap5.com A +short     → 210.109.14.25 확인
3  dig @8.8.8.8 captcha.catchap5.com A +short            → 여기까지 뜨면
4  Caddy 블록 추가 → 인증서 자동 발급
5  curl -s https://captcha.catchap5.com/api/config        → 200 확인
```

5번에서 이게 나와야 정상입니다.

```json
{"siteKey":"site_wz4xkko3e4rI8LFYf0KBsLpTmDQfP2a2",
 "embedOrigins":["https://www.catchap5.com"]}
```

참고로 CAA 레코드가 없어서 인증서 발급을 막는 제한은 없습니다.

---

## 스모크 테스트 결과 — 이게 중요합니다

```
HTTP 200 {"success":false,"error":"invalid_or_used_token"}
```

**401 이 아니라 200 이 나온 게 핵심입니다. 시크릿이 맞다는 뜻입니다.**
어제까지 제가 401 에서 막혀 있던 구간이 이걸로 뚫렸습니다.

가짜 토큰이라 `success:false` 인 게 정상이고, 진짜 토큰이면 `true` 가 나옵니다.
**규약 왕복이 처음으로 완결됐습니다. 백엔드 쪽은 손댈 게 없습니다.**

---

## 나머지 — 동의합니다

**단계적 전환** 순서가 맞습니다. 호스트가 실제로 뜬 뒤에 env 를 바꾸세요.
지금 `/captcha-api` 가 정상 동작 중이니 그대로 두시면 됩니다.

`/captcha-api` 는 **위젯 왕복이 한 번 성공한 뒤에** 정리하는 걸로 민서와
이야기해뒀습니다. 그 전에 지우면 롤백할 곳이 없어집니다.

**`expires_in` 그대로 쓰기** 도 동의합니다. 그게 맞습니다.

---

## 사소한 것 두 개

**`MAIN_CAPTCHA_URL` 끝 슬래시** — 제 참조 구현은 `rstrip("/")` 을 해서 괜찮습니다.
그쪽 배포본도 같은지만 확인해주세요. 안 하면 `//api/verify-token` 이 됩니다.

**CSP 값** — `frame-ancestors 'self' https://www.catchap5.com/` 로 적어주셨는데
실제 서버 값은 슬래시가 없습니다. 서버 값이 맞으니 그대로 두시면 됩니다.

---

## 일정

M5 가 8/12 입니다. 위젯이 붙은 뒤에 저희 팀 8명 수집이 들어가야 해서
**DNS + Caddy 블록이 이번 주 안에** 서면 좋겠습니다. 작업 자체는 세 줄입니다.

---
문의: 조성원 (wwdhogo@gmail.com)
