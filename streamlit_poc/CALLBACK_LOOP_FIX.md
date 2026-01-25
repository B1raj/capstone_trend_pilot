# 🔄 OAuth Callback Loop - Quick Fix Guide

## Issue: Callback keeps running and access token not retrieved

---

## ✅ **Step 1: Enable Debug Mode**

Add `?debug=true` to your Streamlit Cloud URL:

```
https://your-app-name.streamlit.app/?debug=true
```

This will show you:
- Current configuration
- Authentication status
- Any error messages

---

## ✅ **Step 2: Check Streamlit Cloud Logs**

1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. Click "Manage app" → "Logs"
4. Look for these sections:

```
======================================================================
🔄 EXCHANGING AUTHORIZATION CODE FOR TOKEN
======================================================================
```

**What to check:**
- ✅ Does Client ID show correctly? (not "NOT SET")
- ✅ Does Redirect URI match your Streamlit URL?
- ✅ Is there a "Response Status: 200" message?
- ❌ Do you see any error messages?

---

## ✅ **Step 3: Verify Secrets Match**

### In Streamlit Cloud Secrets:
```toml
LINKEDIN_REDIRECT_URI = "https://your-app-name.streamlit.app/oauth/callback"
```

### In LinkedIn Developer Portal → Auth tab:
```
https://your-app-name.streamlit.app/oauth/callback
```

**They MUST match EXACTLY:**
- ✅ Same protocol (https://)
- ✅ Same domain
- ✅ Same path (/oauth/callback)
- ✅ No trailing slash
- ✅ No extra spaces

---

## 🔍 **Common Errors & Solutions**

### Error: "redirect_uri_mismatch"

**Cause:** LinkedIn Developer Portal and Streamlit Secrets don't match

**Fix:**
1. Check LinkedIn Developer Portal → Auth tab → Redirect URIs
2. Make sure it has: `https://your-app-name.streamlit.app/oauth/callback`
3. Check Streamlit Secrets → `LINKEDIN_REDIRECT_URI`
4. They must be EXACTLY the same

### Error: "invalid_client" or "unauthorized_client"

**Cause:** Client ID or Secret is wrong

**Fix:**
1. Go to LinkedIn Developer Portal → Settings
2. Copy your **Client ID** and **Client Secret**
3. Update Streamlit Cloud Secrets:
   ```toml
   LINKEDIN_CLIENT_ID = "paste_here"
   LINKEDIN_CLIENT_SECRET = "paste_here"
   ```
4. Reboot your app

### Error: "LinkedIn credentials not configured"

**Cause:** Secrets not set in Streamlit Cloud

**Fix:**
1. Go to Streamlit Cloud → Your app → Settings → Secrets
2. Add ALL required variables:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   LINKEDIN_CLIENT_ID = "your_id"
   LINKEDIN_CLIENT_SECRET = "your_secret"
   LINKEDIN_REDIRECT_URI = "https://your-app-name.streamlit.app/oauth/callback"
   ```
3. Save and reboot

### Error: "No access token received from LinkedIn"

**Cause:** Token exchange succeeded but didn't return an access token

**Fix:**
1. Check LinkedIn Developer Portal → Products
2. Make sure you have access to "Sign In with LinkedIn using OpenID Connect"
3. If not, click "Request access"
4. Wait for approval (usually instant for basic OAuth)

---

## 📊 **Verification Checklist**

Before testing again, verify:

- [ ] **Streamlit Cloud Secrets are set** (5 variables)
- [ ] **Secrets saved and app rebooted**
- [ ] **LinkedIn Portal has HTTPS redirect URI**
- [ ] **Redirect URIs match EXACTLY**
- [ ] **Scopes enabled:** openid, profile, email
- [ ] **LinkedIn app is active** (not disabled)
- [ ] **Products approved:** Sign In with LinkedIn using OpenID Connect

---

## 🔬 **Debug Steps to Try**

### 1. Check Logs in Real-Time

1. Open Streamlit Cloud logs
2. Click "Connect LinkedIn Account" in your app
3. Watch the logs for:
   ```
   🔐 OAUTH CALLBACK HANDLER
   🔄 EXCHANGING AUTHORIZATION CODE FOR TOKEN
   ```
4. Look for error messages

### 2. Test with Debug Mode

```
https://your-app-name.streamlit.app/?debug=true
```

1. Click "Connect LinkedIn Account"
2. After redirect, check the Debug panel
3. Look for **Auth Error** field
4. Note the exact error message

### 3. Clear Session State

In debug mode:
1. Click "🗑️ Clear All Session State"
2. Try authentication again
3. Watch for different behavior

---

## 💡 **Expected Behavior**

When OAuth works correctly, you should see in logs:

```
======================================================================
🔐 OAUTH CALLBACK HANDLER
======================================================================
Processing OAuth callback with code: AQT...

======================================================================
🔄 EXCHANGING AUTHORIZATION CODE FOR TOKEN
======================================================================
Client ID: 12345...
Redirect URI: https://your-app.streamlit.app/oauth/callback
Authorization Code: AQT...

Sending token exchange request to LinkedIn...
Response Status: 200
✅ Successfully received access token
Token type: Bearer
Expires in: 5184000 seconds
======================================================================

✅ Successfully authenticated! Access token received.
```

Then the app should show:
```
✅ Successfully authenticated with LinkedIn!
```
With confetti/balloons animation.

---

## 🆘 **Still Not Working?**

### Share These Details:

1. **Exact error from debug mode** (`?debug=true`)
2. **Log output** from Streamlit Cloud (copy the OAuth sections)
3. **Confirmation:**
   - ✅ Redirect URI in Portal: `https://your-app-name.streamlit.app/oauth/callback`
   - ✅ Redirect URI in Secrets: `https://your-app-name.streamlit.app/oauth/callback`
   - ✅ They match exactly
   - ✅ Client ID and Secret are set correctly

---

## 🎯 **Quick Test**

Try this URL directly (replace with your details):

```
https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id=YOUR_CLIENT_ID&redirect_uri=https://your-app-name.streamlit.app/oauth/callback&scope=openid%20profile%20email&state=test123
```

This bypasses the app and goes directly to LinkedIn.

**Expected:**
1. LinkedIn login/authorization page
2. After clicking "Allow", redirected to your app
3. URL will have `?code=...`
4. App should process the code

If this doesn't work, the issue is with LinkedIn configuration, not the app code.

---

**Last Updated:** 2026-01-10

**Related Docs:** STREAMLIT_CLOUD_SETUP.md
