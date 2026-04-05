<template>
  <div id="app">
    <!-- 导航栏 -->
    <el-header style="text-align: center; font-size: 20px">
      <el-menu
        :default-active="activePath"
        mode="horizontal"
        background-color="#1989fa"
        text-color="#fff"
        active-text-color="#ffd04b"
        :router="true"
      >
        <el-menu-item index="/">
          <el-icon><House /></el-icon> 首页
        </el-menu-item>
        <el-menu-item index="/single-query">
          <el-icon><Search /></el-icon> 单次查询
        </el-menu-item>
        <el-menu-item index="/batch-analysis">
          <el-icon><Upload /></el-icon> 批量分析
        </el-menu-item>
        <el-menu-item index="/history-record">
          <el-icon><Clock /></el-icon> 历史记录
        </el-menu-item>
      </el-menu>
    </el-header>

    <!-- 路由视图 -->
    <el-main>
      <router-view />
    </el-main>

    <!-- 页脚 -->
    <el-footer style="text-align: center">
      河南师范大学 齐旭鑫 中文评论情感分析系统
    </el-footer>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { House, Search, Upload, Clock } from '@element-plus/icons-vue'

const route = useRoute()
const activePath = ref(route.path)

// 监听路由变化，更新导航激活状态
watch(
  () => route.path,
  (newPath) => {
    activePath.value = newPath
  }
)
</script>

<style scoped>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}
.el-header {
  background-color: #1989fa;
  color: #333;
  line-height: 60px;
}
.el-footer {
  background-color: #f5f5f5;
  color: #666;
  line-height: 60px;
}
.el-main {
  padding: 20px;
}
</style>