<template>
  <div class="batch-analysis-container">
    <el-card shadow="hover">
      <h2 class="card-title">批量TXT文件情感分析</h2>
      <div class="upload-area">
        <el-upload
          class="upload-demo"
          drag
          action="#"
          :auto-upload="false"
          :on-change="handleFileChange"
          :file-list="fileList"
          accept=".txt"
        >
          <i class="el-icon-upload"></i>
          <div class="el-upload__text">
            将TXT文件拖到此处，或<em>点击上传</em>（文件要求：每行一条中文评论）
          </div>
          <div class="el-upload__tip" slot="tip">
            仅支持.txt格式文件，文件内容需为纯文本，每行一条待分析评论
          </div>
        </el-upload>
      </div>

      <el-button
        type="primary"
        @click="handleBatchAnalysis"
        :loading="loading"
        :disabled="!fileList.length"
      >
        <el-icon><Refresh /></el-icon> 开始批量分析
      </el-button>

      <!-- 分析结果展示 -->
      <el-divider content-position="left" v-if="analysisResult.length">分析结果</el-divider>
      <el-table
        v-if="analysisResult.length"
        :data="analysisResult"
        border
        style="width: 100%; margin-top: 20px"
      >
        <el-table-column prop="text" label="评论文本" min-width="500"></el-table-column>
        <el-table-column prop="sentiment_desc" label="情感倾向">
          <template #default="scope">
            <el-tag :type="scope.row.sentiment === '1' ? 'success' : 'danger'">
              {{ scope.row.sentiment_desc }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>

      <!-- 下载报告按钮 -->
      <el-button
        type="success"
        @click="downloadReport"
        :disabled="!analysisResult.length"
        style="margin-top: 20px"
      >
        <el-icon><Download /></el-icon> 下载分析报告
      </el-button>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { predictSentiment } from '../utils/request'
import { Refresh, Download } from '@element-plus/icons-vue'
import fileDownload from 'js-file-download'

// 上传的文件列表
const fileList = ref([])
// 加载状态
const loading = ref(false)
// 批量分析结果
const analysisResult = ref([])

// 处理文件上传
const handleFileChange = (file) => {
  fileList.value = [file] // 只保留最新上传的一个文件
  analysisResult.value = [] // 清空之前的分析结果
}

// 读取TXT文件内容
const readTxtFile = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      // 按行分割文本（兼容Windows/Mac换行符）
      const textContent = e.target.result
      const lines = textContent.split(/\r?\n/).filter(line => line.trim()) // 过滤空行
      resolve(lines)
    }
    reader.onerror = (err) => reject(err)
    reader.readAsText(file.raw, 'utf-8')
  })
}

// 批量分析逻辑
const handleBatchAnalysis = async () => {
  loading.value = true
  analysisResult.value = []
  try {
    // 读取文件内容
    const lines = await readTxtFile(fileList.value[0])
    if (!lines.length) {
      ElMessage.warning('文件内容为空，请检查！')
      return
    }

    // 逐行调用接口分析
    const resultList = []
    for (const line of lines) {
      try {
        const res = await predictSentiment(line)
        resultList.push(res.data)
      } catch (err) {
        // 单条失败不中断整体流程
        resultList.push({
          text: line,
          sentiment: 'error',
          sentiment_desc: '分析失败'
        })
        console.error(`分析文本【${line}】失败：`, err)
      }
    }
    analysisResult.value = resultList

    // 保存批量分析记录到历史
    saveToHistory({
      type: 'batch',
      content: `批量分析${lines.length}条评论（文件：${fileList.value[0].name}）`,
      result: `成功${resultList.filter(item => item.sentiment !== 'error').length}条，失败${resultList.filter(item => item.sentiment === 'error').length}条`,
      time: new Date().toLocaleString(),
      detail: resultList
    })
  } catch (err) {
    ElMessage.error('文件读取失败：' + err.message)
  } finally {
    loading.value = false
  }
}

// 保存历史记录
const saveToHistory = (record) => {
  const history = JSON.parse(localStorage.getItem('sentimentHistory') || '[]')
  history.unshift(record)
  localStorage.setItem('sentimentHistory', JSON.stringify(history))
}

// 下载分析报告
const downloadReport = () => {
  // 生成CSV格式报告
  let csvContent = '评论文本,情感倾向\n'
  analysisResult.value.forEach(item => {
    // 处理文本中的逗号和换行符
    const text = item.text.replace(/,/g, '，').replace(/\n/g, ' ')
    csvContent += `${text},${item.sentiment_desc}\n`
  })
  // 下载文件
  fileDownload(csvContent, `情感分析报告_${new Date().getTime()}.csv`, 'text/csv')
}
</script>

<style scoped>
.batch-analysis-container {
  width: 80%;
  margin: 20px auto;
}
.card-title {
  color: #1989fa;
  margin-bottom: 20px;
}
.upload-area {
  margin-bottom: 20px;
}
</style>