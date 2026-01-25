# 🌐 Streamlit Cloud Deployment Guide

## OAuth Configuration for Streamlit Cloud

When deploying to Streamlit Cloud, you need to update your LinkedIn OAuth configuration.

---

## 📋 Step-by-Step Setup

### 1. Find Your Streamlit Cloud URL

After deploying, your app URL will be:
```
https://your-app-name.streamlit.app
```

Example:
```
https://linkedin-post-generator.streamlit.app
```

### 2. Configure LinkedIn Developer Portal

1. Go to https://www.linkedin.com/developers/apps
2. Select your LinkedIn app
3. Navigate to **"Auth"** tab
4. Under **"OAuth 2.0 settings"**, add a new redirect URI:

   ```
   https://your-app-name.streamlit.app/oauth/callback
   ```

   **Important:**
   - ✅ Must use **HTTPS** (not http)
   - ✅ No trailing slash
   - ✅ Replace `your-app-name` with your actual app name
   - ✅ Keep the `/oauth/callback` path

5. **Keep your localhost URI** for local development:
   ```
   http://localhost:8501/oauth/callback
   ```

6. Click **"Update"** to save

### 3. Configure Streamlit Cloud Secrets

1. Go to https://share.streamlit.io
2. Click on your app
3. Click **"Settings"** (⚙️ icon)
4. Click **"Secrets"**
5. Add the following in TOML format:

```toml
# Required API Keys
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."

# LinkedIn OAuth Credentials
LINKEDIN_CLIENT_ID = "your_client_id_here"
LINKEDIN_CLIENT_SECRET = "your_client_secret_here"
LINKEDIN_REDIRECT_URI = "https://your-app-name.streamlit.app/oauth/callback"

# Optional - Twitter/X API
X_API_KEY = "your_x_bearer_token_here"
```

**Important:**
- Replace `your-app-name` with your actual Streamlit app name
- Use your actual API keys and LinkedIn credentials
- Make sure `LINKEDIN_REDIRECT_URI` matches exactly what you added in LinkedIn Developer Portal

6. Click **"Save"**

### 4. Redeploy Your App

After saving secrets:
1. Go back to your app
2. Click **"Reboot app"** or wait for automatic reboot
3. App will restart with new configuration

---

## 🐛 Troubleshooting

### Issue: OAuth Callback Loop (Keeps Loading)

**Symptoms:**
- After LinkedIn authorization, page keeps loading
- URL shows `?code=...` but nothing happens
- App seems stuck

**Solutions:**

1. **Check Redirect URI Match**
   - Ensure LinkedIn Developer Portal has: `https://your-app-name.streamlit.app/oauth/callback`
   - Ensure Streamlit Secrets has: `LINKEDIN_REDIRECT_URI = "https://your-app-name.streamlit.app/oauth/callback"`
   - They MUST match exactly

2. **Verify Secrets are Set**
   - Check Streamlit Cloud → Settings → Secrets
   - Ensure all required variables are present
   - Redeploy after adding secrets

3. **Enable Debug Mode**
   - Add `?debug=true` to your app URL:
     ```
     https://your-app-name.streamlit.app/?debug=true
     ```
   - This will show debug information to help diagnose the issue

4. **Check Logs**
   - In Streamlit Cloud, click "Manage app" → "Logs"
   - Look for errors like:
     - "Failed to exchange code for token"
     - "Invalid client credentials"
     - "Redirect URI mismatch"

### Issue: "Invalid Redirect URI"

**Solution:**
LinkedIn Developer Portal redirect URI must EXACTLY match:
- Protocol: `https://` (not `http://`)
- Domain: Your Streamlit app domain
- Path: `/oauth/callback`
- No trailing slash

### Issue: "Invalid Client Credentials"

**Solution:**
1. Verify secrets in Streamlit Cloud are correct
2. Check you copied the full Client ID and Secret from LinkedIn
3. Ensure no extra spaces or quotes in secrets

### Issue: "Can't Access Secrets"

**Solution:**
Streamlit Cloud reads secrets from `st.secrets`, but our code uses `os.getenv()`.

Add this to the top of `config.py` (already included):
```python
import os
from dotenv import load_dotenv

load_dotenv(override=False)

# Read from environment (works both locally and on Streamlit Cloud)
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
```

Streamlit Cloud automatically exposes secrets as environment variables, so this works!

---

## ✅ Verification Checklist

Before testing OAuth on Streamlit Cloud:

- [ ] App deployed to Streamlit Cloud
- [ ] LinkedIn Developer Portal has HTTPS redirect URI added
- [ ] Streamlit Cloud secrets configured (all 5 variables)
- [ ] App rebooted after adding secrets
- [ ] Redirect URIs match exactly (Portal vs Secrets)
- [ ] LinkedIn app is active (not disabled)
- [ ] Required scopes enabled: `openid`, `profile`, `email`

---

## 🔄 Testing the OAuth Flow

1. **Open your Streamlit Cloud app**
   ```
   https://your-app-name.streamlit.app
   ```

2. **Click "Connect LinkedIn Account"**
   - You'll be redirected to LinkedIn
   - URL will be: `https://www.linkedin.com/oauth/v2/authorization?...`

3. **Authorize on LinkedIn**
   - Click "Allow" to grant permissions
   - You'll be redirected back to your app

4. **Verify Success**
   - You should see: "✅ Successfully authenticated with LinkedIn!"
   - Your profile should load (name, email)
   - You can now generate posts

---

## 🔐 Security Notes

### Secrets Management

- ✅ Secrets are encrypted in Streamlit Cloud
- ✅ Never commit `.env` file to Git
- ✅ Use `.env.example` as a template only
- ✅ Each developer should have their own `.env` file locally

### Production Best Practices

1. **Use separate LinkedIn apps** for development and production
2. **Rotate secrets** regularly
3. **Monitor API usage** in LinkedIn Developer Portal
4. **Set up CORS** if needed for additional security
5. **Use environment-specific secrets**

---

## 📊 Environment Variables Summary

| Variable | Required | Local Dev | Streamlit Cloud |
|----------|----------|-----------|-----------------|
| `OPENAI_API_KEY` | ✅ Yes | `.env` or shell | Secrets |
| `ANTHROPIC_API_KEY` | ✅ Yes | `.env` or shell | Secrets |
| `LINKEDIN_CLIENT_ID` | ✅ Yes | `.env` or shell | Secrets |
| `LINKEDIN_CLIENT_SECRET` | ✅ Yes | `.env` or shell | Secrets |
| `LINKEDIN_REDIRECT_URI` | ⚠️ Auto | Uses default localhost | **Must set** in Secrets |
| `X_API_KEY` | ❌ Optional | `.env` or shell | Secrets |

---

## 🎯 Quick Reference

### Local Development
```bash
# .env file
LINKEDIN_REDIRECT_URI=http://localhost:8501/oauth/callback
```

### Streamlit Cloud
```toml
# Secrets (TOML format)
LINKEDIN_REDIRECT_URI = "https://your-app-name.streamlit.app/oauth/callback"
```

### LinkedIn Developer Portal
```
# Redirect URIs (add both)
http://localhost:8501/oauth/callback          # For local dev
https://your-app-name.streamlit.app/oauth/callback  # For production
```

---

## 🆘 Still Having Issues?

### Enable Debug Mode
Add `?debug=true` to your URL:
```
https://your-app-name.streamlit.app/?debug=true
```

This will show:
- Current query parameters
- Authentication status
- Access token status
- Configuration values

### Check Logs
1. Streamlit Cloud → Manage app → Logs
2. Look for Python `print()` statements
3. Check for OAuth-related errors

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Redirect URI mismatch" | URIs don't match | Verify Portal vs Secrets match exactly |
| "Invalid client credentials" | Wrong ID/Secret | Check secrets are correct |
| "Code already used" | Callback loop | Fixed with `callback_processed` flag |
| "Failed to exchange token" | Network/API issue | Check logs, verify credentials |

---

## 📞 Support

- **Streamlit Community**: https://discuss.streamlit.io
- **LinkedIn Developer Docs**: https://learn.microsoft.com/linkedin/
- **App Logs**: Check Streamlit Cloud logs for detailed errors

---

**Last Updated**: 2026-01-10

**Compatible with**: Streamlit Cloud, LinkedIn OAuth 2.0
