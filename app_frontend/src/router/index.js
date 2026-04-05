import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import SingleQuery from '../views/SingleQuery.vue'
import BatchAnalysis from '../views/BatchAnalysis.vue'
import HistoryRecord from '../views/HistoryRecord.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/single-query',
    name: 'SingleQuery',
    component: SingleQuery
  },
  {
    path: '/batch-analysis',
    name: 'BatchAnalysis',
    component: BatchAnalysis
  },
  {
    path: '/history-record',
    name: 'HistoryRecord',
    component: HistoryRecord
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router