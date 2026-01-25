# LinkedIn OAuth Setup Guide

## Overview

This application uses LinkedIn OAuth 2.0 to securely access your LinkedIn profile data for generating personalized posts.

## Required Configuration

You must set these environment variables before running the app:

```bash
# Set in your .env file or export in terminal
export LINKEDIN_CLIENT_ID="<your_client_id>"
export LINKEDIN_CLIENT_SECRET="<your_client_secret>"
export LINKEDIN_REDIRECT_URI="http://localhost:8501/oauth/callback"
```

**How to get LinkedIn credentials:**
1. Go to https://www.linkedin.com/developers/apps
2. Create a new app or select an existing one
3. Copy the Client ID and Client Secret from the app settings
4. Set them as environment variables

## LinkedIn Developer Portal Setup

### Step 1: Configure Redirect URI

1. Go to [LinkedIn Developers](https://www.linkedin.com/developers/apps)
2. Find or create your LinkedIn app
3. Click on "Auth" tab
4. Under **OAuth 2.0 settings**, add the following redirect URI:
   ```
   http://localhost:8501/oauth/callback
   ```
5. Click "Update"

### Step 2: Configure Required Scopes

Ensure your app has the following scopes enabled:

**Required Scopes**:
- `openid` - Required for OAuth authentication
- `profile` - Access to basic profile information
- `email` - Access to user's email address

**Optional (for more profile data)**:
- `r_basicprofile` - Read basic profile data (if available)
- `r_liteprofile` - Read lite profile data

### Step 3: Verify App Settings

1. In the LinkedIn app settings, ensure:
   - App is not in "Development" mode (or add your account as a test user)
   - All required scopes are checked
   - Redirect URI exactly matches: `http://localhost:8501/oauth/callback`

## Running the OAuth-Enabled App

### Start the Application

```bash
# Use the OAuth-enabled version
streamlit run app_oauth.py --server.port 8501
```

Or use the startup script:

```bash
# Create a new startup script
./start_oauth_app.sh
```

### OAuth Flow

1. **Open the app**: http://localhost:8501
2. **Click "Connect LinkedIn Account"**
3. **Authorize on LinkedIn**: You'll be redirected to LinkedIn
4. **Grant permissions**: Allow the app to access your profile
5. **Redirected back**: You'll return to the app with authentication complete
6. **Generate posts**: Now you can generate personalized LinkedIn posts!

## Troubleshooting

### Issue: "Redirect URI mismatch"

**Solution**: Ensure the redirect URI in LinkedIn Developer Portal exactly matches:
```
http://localhost:8501/oauth/callback
```

Note: No trailing slash, exact port number (8501)

### Issue: "Invalid client credentials"

**Solution**:
1. Verify Client ID and Client Secret in `config.py`
2. Check that the app is active in LinkedIn Developer Portal
3. Ensure you're using the correct credentials

### Issue: "Insufficient permissions"

**Solution**:
1. Check that all required scopes are enabled in LinkedIn Developer Portal
2. Scopes needed: `openid`, `profile`, `email`
3. Re-authenticate after adding new scopes

### Issue: "Authentication successful but can't fetch profile"

**Solution**:
1. LinkedIn's basic OAuth only provides limited profile data
2. For detailed profile data (headline, about, skills), you may need to apply for additional API access from LinkedIn
3. Current implementation uses basic userinfo which provides: name, email, picture

## API Limitations

### What OAuth Provides

With standard LinkedIn OAuth (`openid`, `profile`, `email` scopes):
- ✅ Full name
- ✅ Email address
- ✅ Profile picture
- ✅ User ID (sub)
- ✅ Locale

### What Requires Additional Access

These fields require LinkedIn Partner Program or additional API access:
- ❌ Headline
- ❌ About/Summary section
- ❌ Experience/Work history
- ❌ Skills
- ❌ Connections

### Workaround

For testing purposes, the app will:
1. Use OAuth to authenticate and get basic profile (name, email)
2. You can manually enter additional profile info (headline, skills) in the UI
3. Or we can use placeholder data for testing

## Alternative: Enhanced Profile Access

To get full profile access, you need to:

1. **Join LinkedIn Partner Program**: Apply at https://www.linkedin.com/developers/apps
2. **Request Marketing Developer Platform access**
3. **Get approved for additional scopes**:
   - `r_basicprofile`
   - `r_liteprofile`
   - `r_emailaddress`

This process can take several weeks and requires business justification.

## For Development/Testing

If you can't access full profile data via OAuth:

### Option 1: Manual Profile Input

We can add a form where users enter:
- Headline
- Brief about section
- Key skills (comma-separated)

### Option 2: Use Sample Data

For testing, use pre-populated sample profiles.

### Option 3: Hybrid Approach (Recommended)

1. Authenticate with OAuth (gets name, email)
2. Ask user to paste their LinkedIn profile URL
3. Extract additional data from the public profile HTML (fallback to scraping)

## Security Notes

- Never commit Client Secret to version control
- Store credentials in environment variables or .env file
- Use HTTPS in production (not http://localhost)
- Implement state parameter for CSRF protection (already included)

## Production Deployment

For production use (not localhost):

1. Update redirect URI to your production domain:
   ```
   https://yourdomain.com/oauth/callback
   ```

2. Add this URI to LinkedIn Developer Portal

3. Update `config.py`:
   ```python
   LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "https://yourdomain.com/oauth/callback")
   ```

4. Use HTTPS for all OAuth flows

## Testing Checklist

- [ ] Redirect URI configured in LinkedIn Developer Portal
- [ ] All required scopes enabled
- [ ] App running on port 8501
- [ ] Can click "Connect LinkedIn Account"
- [ ] Successfully redirected to LinkedIn
- [ ] Can authorize the app
- [ ] Successfully redirected back to app
- [ ] Profile data loaded (at minimum: name and email)
- [ ] Can generate posts with authenticated profile

## Current Status

✅ OAuth handler implemented
✅ OAuth flow integrated in Streamlit app
✅ Config updated with credentials
⚠️ **Action Required**: Configure redirect URI in LinkedIn Developer Portal
⚠️ **Note**: Full profile access may be limited without Partner Program access

## Next Steps

1. Configure redirect URI in LinkedIn Developer Portal
2. Test OAuth authentication flow
3. Verify profile data retrieval
4. Generate first LinkedIn post!

---

**Need Help?**
- LinkedIn OAuth Documentation: https://docs.microsoft.com/en-us/linkedin/shared/authentication/authentication
- LinkedIn Developer Support: https://www.linkedin.com/developers/support
