# Cloudflare Images Setup Guide

This guide shows you how to set up Cloudflare Images for cloud-based image storage with your adaptive camera sampling renderer.

## 1. Enable Cloudflare Images

### Step 1: Access Cloudflare Dashboard
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Select your account
3. Navigate to **Images** in the left sidebar

### Step 2: Enable Cloudflare Images
1. Click **"Get Started"** on the Images page
2. Choose your pricing plan:
   - **Free tier**: 100,000 images, 500,000 transformations/month
   - **Paid tier**: $5/month for 100,000 images + $1 per additional 1,000 images

## 2. Get API Credentials

### Step 1: Get Account ID
1. In Cloudflare Dashboard, go to the right sidebar
2. Copy your **Account ID** (it looks like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`)

### Step 2: Create API Token
1. Go to **My Profile** → **API Tokens**
2. Click **"Create Token"**
3. Use **"Custom token"** template
4. Configure permissions:
   - **Account**: `Cloudflare Images:Edit`
   - **Zone Resources**: `Include - All zones` (or specific zones)
5. Click **"Continue to summary"**
6. Click **"Create Token"**
7. **Copy and save the token** (you won't see it again!)

## 3. Test Your Setup

### Option A: Using the Test Script
```bash
python test_api_workflow.py \
  --url http://localhost:5000 \
  --cloudflare-account-id YOUR_ACCOUNT_ID \
  --cloudflare-api-token YOUR_API_TOKEN
```

### Option B: Using curl
```bash
# Test API endpoint directly
curl -X POST http://localhost:5000/sample_render \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "num_samples": 1,
    "return_albedo": true,
    "cloud_storage": "cloudflare",
    "embed_images": false,
    "cloudflare_account_id": "YOUR_ACCOUNT_ID",
    "cloudflare_api_token": "YOUR_API_TOKEN"
  }'
```

## 4. Example Usage in Code

### Python Example
```python
import requests

# Setup renderer session
setup_response = requests.post('http://localhost:5000/setup_renderer', json={
    "render_width": 512,
    "render_height": 512,
    "use_nf_sampler": True
})
session_id = setup_response.json()['session_id']

# Render with Cloudflare Images storage
render_response = requests.post('http://localhost:5000/sample_render', json={
    "session_id": session_id,
    "num_samples": 5,
    "return_albedo": True,
    "cloud_storage": "cloudflare",
    "embed_images": False,
    "cloudflare_account_id": "your_account_id_here",
    "cloudflare_api_token": "your_api_token_here"
})

job_id = render_response.json()['job_id']

# Check results
import time
while True:
    status_response = requests.get(f'http://localhost:5000/status/{job_id}')
    status = status_response.json()

    if status['status'] == 'completed':
        results = status['results']
        for i, result in enumerate(results):
            print(f"Sample {i}: {result['albedo_url']}")
        break
    elif status['status'] == 'error':
        print(f"Error: {status['error']}")
        break

    time.sleep(2)
```

### JavaScript Example
```javascript
// Setup renderer session
const setupResponse = await fetch('http://localhost:5000/setup_renderer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        render_width: 512,
        render_height: 512,
        use_nf_sampler: true
    })
});
const { session_id } = await setupResponse.json();

// Render with Cloudflare Images storage
const renderResponse = await fetch('http://localhost:5000/sample_render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        session_id: session_id,
        num_samples: 5,
        return_albedo: true,
        cloud_storage: 'cloudflare',
        embed_images: false,
        cloudflare_account_id: 'your_account_id_here',
        cloudflare_api_token: 'your_api_token_here'
    })
});
const { job_id } = await renderResponse.json();

// Check results
const checkStatus = async () => {
    const statusResponse = await fetch(`http://localhost:5000/status/${job_id}`);
    const status = await statusResponse.json();

    if (status.status === 'completed') {
        status.results.forEach((result, i) => {
            console.log(`Sample ${i}: ${result.albedo_url}`);
        });
    } else if (status.status === 'error') {
        console.error(`Error: ${status.error}`);
    } else {
        setTimeout(checkStatus, 2000);
    }
};
checkStatus();
```

## 5. Storage Options Comparison

| Storage Type | Pros | Cons | Use Case |
|--------------|------|------|----------|
| **Base64 Embedding** | ✅ No setup required<br>✅ Works immediately<br>✅ No external dependencies | ❌ Large response sizes<br>❌ Not suitable for many images<br>❌ No caching | Quick testing, small batches |
| **Cloudflare Images** | ✅ Global CDN<br>✅ Automatic optimization<br>✅ Permanent URLs<br>✅ Built-in transformations | ❌ Requires setup<br>❌ Costs money<br>❌ API rate limits | Production use, large batches |
| **Local File + ngrok** | ✅ Free<br>✅ Full control | ❌ Temporary URLs<br>❌ Single server<br>❌ No optimization | Development, temporary sharing |

## 6. Troubleshooting

### Common Issues

**"Cloudflare credentials not configured"**
- Make sure you're passing both `cloudflare_account_id` and `cloudflare_api_token`
- Verify the credentials are correct

**"HTTP 403: Forbidden"**
- Check your API token permissions
- Ensure the token has `Cloudflare Images:Edit` permission
- Verify the account ID is correct

**"HTTP 429: Too Many Requests"**
- You've hit Cloudflare's rate limits
- Wait a few minutes and try again
- Consider reducing the number of concurrent uploads

**Images not appearing**
- Check the returned URL in a browser
- Verify the image was actually uploaded to Cloudflare
- Check Cloudflare Images dashboard for uploaded images

### Debug Mode
Enable verbose logging to see detailed upload information:
```python
# In your renderer setup
renderer = BasicRasterizer(verbose=True)
```

## 7. Advanced Configuration

### Custom Image Transformations
Cloudflare Images supports URL-based transformations:
```
# Original image
https://imagedelivery.net/your-account-hash/image-id/public

# Resized to 300x300
https://imagedelivery.net/your-account-hash/image-id/w=300,h=300

# Converted to WebP
https://imagedelivery.net/your-account-hash/image-id/format=webp
```

### Batch Upload Optimization
For large batches, consider:
1. Using smaller image quality settings (70-80% JPEG quality)
2. Implementing retry logic for failed uploads
3. Adding upload progress tracking
4. Using concurrent uploads (with rate limiting)

## 8. Security Best Practices

1. **Never expose API tokens in client-side code**
2. **Use environment variables for credentials**
3. **Implement token rotation**
4. **Monitor API usage in Cloudflare dashboard**
5. **Set up IP restrictions if possible**

### Environment Variables Example
```bash
# .env file
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_API_TOKEN=your_api_token_here
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

cloudflare_credentials = {
    "account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    "api_token": os.getenv("CLOUDFLARE_API_TOKEN")
}
```

---

## Need Help?

- **Cloudflare Images Documentation**: https://developers.cloudflare.com/images/
- **API Reference**: https://developers.cloudflare.com/api/operations/cloudflare-images-upload-an-image-via-url
- **Community Support**: https://community.cloudflare.com/

Happy rendering! 🎨📸
