# CVSLR Deployment Guide

## Deploy to Render (Free)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration
   - Click "Create Web Service"

3. **Wait for Build**
   - First build takes 5-10 minutes
   - Render will install all dependencies
   - Your app will be live at: `https://cvslr-app.onrender.com`

### Important Notes

⚠️ **Free Tier Limitations:**
- App sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds to wake up
- 512MB RAM limit
- No GPU support

💡 **Tips:**
- Share the URL with others to view your demo
- App works best with smaller video files
- Live camera may be slower on free tier

### Alternative: Deploy to Hugging Face Spaces

If Render is too slow, consider Hugging Face Spaces:
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space
3. Upload your code
4. Get permanent public URL

### Troubleshooting

**Build fails?**
- Check that all model files are in the repo
- Ensure `best_cnn_transformer_hybrid.pth` is < 100MB
- Verify `.task` files are included

**App crashes?**
- Reduce video resolution in `app.py`
- Lower frame rate to reduce CPU usage
- Check Render logs for errors

**Too slow?**
- Upgrade to Render paid plan ($7/month)
- Or use Hugging Face Spaces with GPU

---

Need help? Check Render documentation: https://render.com/docs
