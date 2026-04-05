<template>
  <div class="home-container">
    <el-card class="home-card" shadow="hover">
      <h1 class="title">中文评论情感分析系统</h1>
      <div class="desc">
        <p>本系统基于机器学习模型实现中文评论的情感倾向分析（正向/负向），支持单次文本查询、批量TXT文件分析，并提供历史记录管理功能。</p>
      </div>
      <div class="btn-group">
        <el-button type="primary" size="large" @click="$router.push('/single-query')">
          <el-icon><Search /></el-icon> 单次文本查询
        </el-button>
        <el-button type="success" size="large" @click="$router.push('/batch-analysis')">
          <el-icon><Upload /></el-icon> 批量文件分析
        </el-button>
        <el-button type="warning" size="large" @click="$router.push('/history-record')">
          <el-icon><Clock /></el-icon> 查看历史记录
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { Search, Upload, Clock } from '@element-plus/icons-vue'
</script>

<style scoped>
.home-container {
  width: 100%;
  height: 80vh;
  display: flex;
  justify-content: center;
  align-items: center;
}
.home-card {
  width: 80%;
  padding: 40px;
  text-align: center;
}
.title {
  color: #1989fa;
  margin-bottom: 30px;
}
.desc {
  font-size: 16px;
  line-height: 1.8;
  margin-bottom: 40px;
  color: #666;
}
.btn-group {
  display: flex;
  justify-content: center;
  gap: 20px;
}
</style>