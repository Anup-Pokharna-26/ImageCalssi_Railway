# 🚀 Deployment Guide - Indian Food Classifier

## 📋 Pre-deployment Checklist

✅ Model is properly trained (11.17 MB)  
✅ 80 Indian food classes configured  
✅ Requirements.txt optimized for production  
✅ Gunicorn configured for production server  
✅ Environment variables set  
✅ Google Drive integration working  

## 🏆 Recommended Platform: Railway

**Why Railway?**
- ✅ Perfect for ML applications
- ✅ 512MB RAM free tier (sufficient for our 11MB model)  
- ✅ Automatic HTTPS and custom domains
- ✅ No cold start issues
- ✅ Easy GitHub integration

### 🚀 Deploy to Railway (Recommended)

1. **Create Railway Account**
   ```bash
   # Visit: https://railway.app
   # Sign up with GitHub
   ```

2. **Prepare Repository**
   ```bash
   # Ensure all files are committed
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

3. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "Deploy Now"
   - Connect your GitHub repository
   - Select this repository
   - Railway will automatically detect Python and deploy!

4. **Environment Variables (Auto-configured)**
   ```
   FLASK_ENV=production
   FLASK_DEBUG=False
   PYTHONUNBUFFERED=1
   TF_CPP_MIN_LOG_LEVEL=2
   ```

### 🔧 Alternative Platform: Render

1. **Deploy to Render**
   - Visit [render.com](https://render.com)
   - Connect GitHub repository
   - Choose "Web Service"
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`

### 🐳 Alternative Platform: Heroku

1. **Install Heroku CLI**
   ```bash
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### ☁️ Alternative Platform: Google Cloud Run

1. **Build and Deploy**
   ```bash
   gcloud run deploy --source .
   ```

## 🔍 Deployment Verification

After deployment, test these endpoints:

1. **Health Check**
   ```
   GET https://your-app.railway.app/
   ```

2. **Prediction Test**
   ```
   POST https://your-app.railway.app/predict
   # Upload an image file
   ```

3. **Enhanced Prediction Test**
   ```
   POST https://your-app.railway.app/predict_enhanced
   # Upload an image file
   ```

## 📊 Expected Performance

- **Cold Start**: ~3-5 seconds (first request)
- **Warm Requests**: ~200-500ms per prediction
- **Memory Usage**: ~400-600MB
- **Model Load Time**: ~2-3 seconds

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Error**
   ```bash
   # Upgrade to Railway Pro ($5/month) for more RAM
   ```

2. **Model Download Fails**
   ```bash
   # Check Google Drive URLs are public
   # Verify internet connectivity
   ```

3. **TensorFlow Warnings**
   ```bash
   # Already configured: TF_CPP_MIN_LOG_LEVEL=2
   ```

4. **Slow Response Times**
   ```bash
   # Model is loaded once at startup (not per request)
   # First request may be slower due to model initialization
   ```

## 🚀 Production Optimizations

### Current Optimizations
- ✅ Gunicorn with optimized worker configuration
- ✅ Model preloading to avoid cold starts
- ✅ Smart caching of predictions
- ✅ Optimized image preprocessing
- ✅ Error handling and fallbacks

### Performance Settings
```python
# Gunicorn configuration
workers = 1  # Single worker for ML models
max_requests = 100  # Restart worker after 100 requests
timeout = 120  # 2-minute timeout for ML processing
preload_app = True  # Preload model at startup
```

## 📈 Monitoring and Maintenance

### Health Monitoring
- Railway provides built-in monitoring
- Check logs: `railway logs`
- Monitor memory usage in Railway dashboard

### Model Updates
- Update Google Drive model file
- Railway will auto-reload within 5 minutes
- Or trigger manual restart in Railway dashboard

## 💰 Cost Estimation

### Railway (Recommended)
- **Free Tier**: $0/month (512MB RAM, enough for this app)
- **Pro Tier**: $5/month (up to 8GB RAM, if needed)

### Render
- **Free Tier**: $0/month (512MB RAM, sleeps after 15min)
- **Starter**: $7/month (no sleep)

### Heroku
- **Free Tier**: Discontinued
- **Basic**: $7/month (512MB RAM)

## 🎯 Deployment Success Indicators

✅ App loads without errors  
✅ Model predictions working (test with sample food images)  
✅ All 3 prediction endpoints functional  
✅ Response times under 5 seconds  
✅ Memory usage stable around 400-600MB  

## 📞 Support

If you encounter issues:
1. Check Railway/platform logs
2. Verify all environment variables
3. Test locally first: `python app.py`
4. Check model file accessibility

---

**🎉 Ready to deploy! Railway is recommended for the best experience.**
