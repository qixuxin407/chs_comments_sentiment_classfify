<template>
  <div class="history-record-container">
    <el-card shadow="hover">
      <h2 class="card-title">分析历史记录</h2>

      <el-button type="danger" @click="clearAllHistory" :disabled="!historyList.length">
        <el-icon><Delete /></el-icon> 清空所有记录
      </el-button>

      <el-table
        :data="historyList"
        border
        style="width: 100%; margin-top: 20px"
        v-loading="loading"
      >
        <el-table-column prop="time" label="操作时间" width="200"></el-table-column>
        <el-table-column prop="type" label="操作类型">
          <template #default="scope">
            <el-tag :type="scope.row.type === 'single' ? 'primary' : 'success'">
              {{ scope.row.type === 'single' ? '单次查询' : '批量分析' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="content" label="操作内容" min-width="400"></el-table-column>
        <el-table-column prop="result" label="分析结果" min-width="200"></el-table-column>
        <el-table-column label="操作">
          <template #default="scope">
            <el-button
              type="text"
              @click="deleteSingleHistory(scope.row)"
              icon="Delete"
              text-color="danger"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 空状态 -->
      <div class="empty-state" v-if="!historyList.length && !loading">
        <el-empty description="暂无历史记录"></el-empty>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Delete } from '@element-plus/icons-vue'

// 加载状态
const loading = ref(false)
// 历史记录列表
const historyList = ref([])

// 加载历史记录
const loadHistory = () => {
  loading.value = true
  try {
    const history = JSON.parse(localStorage.getItem('sentimentHistory') || '[]')
    historyList.value = history
  } catch (err) {
    ElMessage.error('加载历史记录失败：' + err.message)
  } finally {
    loading.value = false
  }
}

// 删除单条记录
const deleteSingleHistory = (record) => {
  ElMessageBox.confirm('确定要删除这条记录吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    historyList.value = historyList.value.filter(item => item.time !== record.time)
    localStorage.setItem('sentimentHistory', JSON.stringify(historyList.value))
    ElMessage.success('删除成功！')
  })
}

// 清空所有记录
const clearAllHistory = () => {
  ElMessageBox.confirm('确定要清空所有历史记录吗？此操作不可恢复！', '警告', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'error'
  }).then(() => {
    historyList.value = []
    localStorage.removeItem('sentimentHistory')
    ElMessage.success('清空成功！')
  })
}

// 页面挂载时加载历史记录
onMounted(() => {
  loadHistory()
})
</script>

<style scoped>
.history-record-container {
  width: 80%;
  margin: 20px auto;
}
.card-title {
  color: #1989fa;
  margin-bottom: 20px;
}
.empty-state {
  margin-top: 40px;
  text-align: center;
}
</style>