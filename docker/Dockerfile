# ====== 构建阶段（基于 MCR：Ubuntu + OpenJDK17）======
FROM mcr.microsoft.com/openjdk/jdk:17-ubuntu AS build

WORKDIR /app
RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

# 处理 Gradle Wrapper（避免 CRLF）
COPY gradlew gradlew
COPY gradle gradle
RUN dos2unix gradlew && chmod +x gradlew

COPY settings.gradle settings.gradle
COPY build.gradle build.gradle

# 源码 + 本地 libs（含 onnxruntime-1.22.0.jar）
COPY src src
COPY libs libs
COPY src/main/resources src/main/resources

# 打包 Spring Boot 可执行 JAR（跳过测试更快）
RUN ./gradlew bootJar -x test

# ====== 运行阶段（同样用 MCR 基镜像）======
FROM mcr.microsoft.com/openjdk/jdk:17-ubuntu AS runtime
WORKDIR /app

# ONNX Runtime 需要的本地库
RUN apt-get update && apt-get install -y --no-install-recommends \
      libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/build/libs/*.jar /app/app.jar
ENV JAVA_OPTS=""
EXPOSE 8080
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app/app.jar"]
