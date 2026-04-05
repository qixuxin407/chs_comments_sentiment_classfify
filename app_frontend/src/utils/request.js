// src/utils/request.js 完整修改后代码
import axios from 'axios'
import { ElMessage } from 'element-plus' // 新增：引入ElMessage

// 创建axios实例
const service = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json;charset=utf-8'
  }
})

// 请求拦截器
service.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
service.interceptors.response.use(
  (response) => {
    const res = response.data
    if (res.code !== 200) {
      ElMessage.error(res.msg || '请求失败')
      return Promise.reject(res)
    }
    // 注意：这里去掉成功提示，避免每次查询都弹窗，只在组件里按需提示
    // ElMessage.success(res.msg || '请求成功')
    return res
  },
  (error) => {
    ElMessage.error(error.message || '服务器错误')
    return Promise.reject(error)
  }
)

// 单次情感预测接口
export const predictSentiment = (text) => {
  return service.post('/predict_sentiment', { text })
}

export default service