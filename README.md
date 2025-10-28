# practice
practice of anything, include "Beginning Linux Programming"



| 请求头字段             | 直接访问 Cloud Run (200 OK) | 通过 bot.supra.tw (400 Bad Request) | 差异                                  |
| -------------------- | ------------------------------- | ------------------------------------- | ------------------------------------- |
| 请求行           | GET / HTTP/2                  | GET / HTTP/1.1                        | HTTP 协议版本不同 (HTTP/2 vs HTTP/1.1) |
| :authority / Host | surpass-1097446576443.us-west1.run.app | bot.supra.tw                          | 域名不同                            |
| :scheme / X-Forwarded-Proto | https                         |  $scheme (实际应为 https)          |  协议头信息 (应相同)                   |
| 其他 Accept, Accept-Encoding, User-Agent 等 | 基本一致                          | 基本一致                              |  基本一致                             |