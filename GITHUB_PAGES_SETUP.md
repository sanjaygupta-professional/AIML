# GitHub Pages Setup Instructions

This guide will help you publish your AI/ML Algorithm Playbook as a live website using GitHub Pages.

## 📋 Prerequisites

- You need admin/write access to the repository
- Your Assignment#2 directory is already set up with index.html

## 🚀 Step-by-Step Setup

### Option 1: Enable GitHub Pages via Web Interface (Recommended)

1. **Navigate to Repository Settings**
   - Go to: https://github.com/sanjaygupta-professional/AIML
   - Click on **Settings** tab (top right of repository page)

2. **Access Pages Settings**
   - In the left sidebar, scroll down to **Pages** under "Code and automation"
   - Click on **Pages**

3. **Configure Source**
   - Under "Build and deployment" section:
     - **Source**: Select "Deploy from a branch"
     - **Branch**: Select `main` (or your default branch)
     - **Folder**: Select `/ (root)`
   - Click **Save**

4. **Wait for Deployment**
   - GitHub will automatically build and deploy your site
   - This typically takes 1-3 minutes
   - You'll see a message: "Your site is ready to be published at..."

5. **Access Your Published Site**
   - Once deployed, your site will be available at:
     - **Main page**: `https://sanjaygupta-professional.github.io/AIML/`
     - **Assignment #2**: `https://sanjaygupta-professional.github.io/AIML/Assignment%232/`

### Option 2: Enable via GitHub CLI (If Available)

If you have `gh` CLI installed locally:

```bash
# Enable GitHub Pages from command line
gh api repos/sanjaygupta-professional/AIML/pages \
  --method POST \
  -f source[branch]=main \
  -f source[path]=/
```

## 🔗 Your Published URLs

Once GitHub Pages is enabled:

| Resource | URL |
|----------|-----|
| Repository Home | `https://sanjaygupta-professional.github.io/AIML/` |
| Interactive Playbook | `https://sanjaygupta-professional.github.io/AIML/Assignment%232/` |
| Assignment README | `https://sanjaygupta-professional.github.io/AIML/Assignment%232/README.md` |

## 📤 Sharing with Your Team

Share these links with your group members:
- Mike Etuhoko
- RK
- Other team members

They can access the interactive guide directly in their browsers without needing to clone the repository!

## ✅ Verify Deployment

After enabling GitHub Pages:

1. **Check Actions Tab**
   - Go to the **Actions** tab in your repository
   - You should see a workflow named "pages-build-deployment"
   - Wait for the green checkmark ✓

2. **Visit Your Site**
   - Click on the provided URL or visit the URL mentioned above
   - You should see your interactive AI/ML Algorithm Playbook

3. **Test Navigation**
   - Verify all sections load correctly
   - Check that interactive elements work (accordions, selectable cards, etc.)

## 🔧 Troubleshooting

### Issue: Site shows 404 error
- **Solution**: Wait a few more minutes; initial deployment can take up to 10 minutes
- **Check**: Ensure the branch is `main` and path is `/ (root)` in settings

### Issue: Styles not loading
- **Solution**: The `.nojekyll` file is already created in the repository to prevent Jekyll processing
- **Check**: Verify the file exists in the root directory

### Issue: Can't find Pages in Settings
- **Solution**: You may need repository admin permissions
- **Ask**: Repository owner to enable Pages or grant you admin access

### Issue: Assignment#2 URL shows %23 instead of #
- **Explanation**: This is URL encoding for the `#` character and is normal
- **Alternative**: You can access directly at `https://sanjaygupta-professional.github.io/AIML/Assignment%232/index.html`

## 🎯 Next Steps

After GitHub Pages is live:

1. ✅ Share the URL with your team in your group chat/email
2. ✅ Add the live URL to your Assignment#2/README.md
3. ✅ Use the live site for collaborative planning and progress tracking
4. ✅ Test all interactive features with your team

## 📝 Updating the Site

Whenever you push changes to the `main` branch:
- GitHub Pages will automatically rebuild and redeploy
- Changes typically appear within 1-2 minutes
- No additional action required!

## 🛡️ Privacy Settings

By default, GitHub Pages sites are public. If you need to restrict access:

1. Go to **Settings** → **Pages**
2. Under "Visibility", you can change to private (requires GitHub Pro/Team/Enterprise)
3. For public repositories, the Pages site will always be public

---

**Questions?** Check the [GitHub Pages documentation](https://docs.github.com/en/pages) or contact your repository administrator.

**Created:** November 6, 2025
**Repository:** sanjaygupta-professional/AIML
