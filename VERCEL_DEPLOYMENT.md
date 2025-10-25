# Vercel Deployment Guide

## Overview
This guide will help you deploy your Forest Analysis System to Vercel successfully.

## Project Structure
```
├── api/
│   └── index.py          # Vercel entry point
├── templates/            # HTML templates
├── static/              # CSS and static files
├── app.py               # Main Flask application
├── vercel.json          # Vercel configuration
├── requirements.txt     # Python dependencies
└── test_vercel.py       # Deployment test script
```

## Deployment Steps

### 1. Test Locally
```bash
python test_vercel.py
```
This will verify that your app is ready for deployment.

### 2. Commit and Push to GitHub
```bash
git add .
git commit -m "Fix Vercel deployment configuration"
git push origin main
```

### 3. Deploy on Vercel

#### Option A: Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow the prompts:
# - Link to existing project or create new
# - Set up project settings
# - Deploy
```

#### Option B: Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will automatically detect the Python configuration
5. Click "Deploy"

### 4. Environment Variables (if needed)
If your app needs environment variables:
1. Go to your Vercel project dashboard
2. Navigate to Settings > Environment Variables
3. Add any required variables

## Configuration Details

### vercel.json
- Points to `api/index.py` as the entry point
- Routes all requests to the Flask app
- Sets maximum function duration to 30 seconds

### api/index.py
- Imports the Flask app from the parent directory
- Exports the app as `handler` for Vercel

### requirements.txt
- Includes all necessary dependencies
- Compatible with Vercel's Python runtime

## Troubleshooting

### Common Issues

1. **"No code found" error**
   - Ensure `api/index.py` exists and imports the Flask app correctly
   - Check that `vercel.json` points to the correct entry point

2. **Import errors**
   - Make sure all dependencies are in `requirements.txt`
   - Test locally with `python test_vercel.py`

3. **Timeout errors**
   - The `maxDuration` is set to 30 seconds in `vercel.json`
   - For longer operations, consider using background jobs

4. **Static files not loading**
   - Ensure templates and static files are in the correct directories
   - Check that Flask is configured to serve static files

### Testing Your Deployment

Once deployed, test these endpoints:
- `https://your-app.vercel.app/` - API root
- `https://your-app.vercel.app/interface` - Main interface
- `https://your-app.vercel.app/dashboard` - Dashboard
- `https://your-app.vercel.app/api/analyze` - Analysis endpoint

## Features Included

✅ **Forest Analysis System**
- Real-time forest analysis
- Sustainability metrics calculation
- Interactive mapping interface
- PDF report generation
- Global location support
- Advanced dashboard

✅ **Vercel Optimized**
- Proper entry point configuration
- Optimized for serverless deployment
- Fast cold start times
- Automatic scaling

## Support

If you encounter issues:
1. Check the Vercel deployment logs
2. Test locally with `python test_vercel.py`
3. Verify all files are committed to GitHub
4. Check the Vercel documentation for Python deployments
