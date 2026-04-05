<template>
  <div class="single-query-container">
    <el-card shadow="hover">
      <h2 class="card-title">单次文本情感分析</h2>
      <el-form :model="form" label-width="80px">
        <el-form-item label="输入文本">
          <el-input
            v-model="form.text"
            type="textarea"
            rows="5"
            placeholder="请输入需要分析的中文评论（如：这个产品用起来特别好！）"
          ></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handlePredict" :loading="loading">
            <el-icon><Refresh /></el-icon> 开始分析
          </el-button>
          <el-button @click="form.text = ''">清空输入</el-button>
        </el-form-item>
      </el-form>

      <!-- 分析结果展示 -->
      <el-divider content-position="left">分析结果</el-divider>
      <el-card v-if="result" class="result-card">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="输入文本">{{ result.data.text }}</el-descriptions-item>
          <el-descriptions-item label="情感倾向">
            <el-tag :type="result.data.sentiment === '1' ? 'success' : 'danger'">
              {{ result.data.sentiment_desc }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { predictSentiment } from '../utils/request'
import { Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus' // 引入ElMessage

// 表单数据
const form = ref({
  text: ''
})
// 加载状态
const loading = ref(false)
// 分析结果
const result = ref(null)

// 处理预测逻辑
const handlePredict = async () => {
  if (!form.value.text.trim()) {
    ElMessage.warning('请输入需要分析的文本！')
    return
  }
  loading.value = true
  result.value = null // 每次查询前清空旧结果
  try {
    // 调用后端接口
    const res = await predictSentiment(form.value.text)
    result.value = res
    // 保存到历史记录
    saveToHistory({
      type: 'single',
      content: form.value.text,
      result: res.data.sentiment_desc,
      time: new Date().toLocaleString()
    })
    ElMessage.success('分析成功！') // 成功提示
  } catch (err) {
    console.error('分析失败：', err)
    ElMessage.error('分析失败，请稍后重试') // 失败提示
  } finally {
    loading.value = false
  }
}

// 保存历史记录到localStorage
const saveToHistory = (record) => {
  const history = JSON.parse(localStorage.getItem('sentimentHistory') || '[]')
  history.unshift(record) // 新增记录放在最前面
  localStorage.setItem('sentimentHistory', JSON.stringify(history))
}
</script>

<style scoped>
.single-query-container {
  width: 80%;
  margin: 20px auto;
}
.card-title {
  color: #1989fa;
  margin-bottom: 20px;
}
.result-card {
  margin-top: 20px;
}
</style>