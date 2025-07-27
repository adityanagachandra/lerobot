# LeRobot Fork Daily Updates Dashboard

A modern, dark-themed webpage that displays daily updates from the adityanagachandra/lerobot repository fork. The dashboard automatically pulls data from GitHub and displays new files, current work, and recent activity.

## Features

- **Daily Updates**: Automatically tracks new files and changes uploaded to the repository
- **Current Work Tracking**: Shows active issues and pull requests being worked on
- **Recent Activity Feed**: Displays commits, file changes, and issue updates with filtering
- **Auto-refresh**: Updates data every hour during active hours (6 AM - 11 PM) and daily at midnight
- **Modern Dark UI**: Matte, dark theme with serif fonts for an elegant appearance
- **Real-time Statistics**: Shows counts for new files, commits, and active issues

## How to Use

1. **Open the webpage**: Simply open `index.html` in your web browser
2. **View updates**: The page automatically loads the latest repository data
3. **Refresh manually**: Click the "Refresh" button to get the latest updates
4. **Filter activity**: Use the filter buttons (All, Commits, Files, Issues) to view specific types of activity
5. **Explore links**: Click on any item to view it directly on GitHub

## Sections

### Statistics Cards
- **Last Update**: Shows when the data was last refreshed
- **New Files Today**: Count of files added or modified today
- **Commits Today**: Number of commits made today
- **Active Issues**: Currently open issues in the repository

### New Files & Utilities
Displays recently added or modified files with:
- File names and paths
- Change statistics (additions/deletions)
- Inferred descriptions of what each file likely contains
- Links to view files on GitHub

### Currently Working On
Shows active development work including:
- Open issues with descriptions
- Active pull requests
- Status indicators
- Links to GitHub for full details

### Recent Activity
Comprehensive activity feed showing:
- Recent commits with author information
- File changes and modifications
- Issue updates and discussions
- Filterable by activity type

## Technical Details

- **Data Source**: GitHub API for adityanagachandra/lerobot repository
- **Update Frequency**: 
  - Hourly during active hours (6 AM - 11 PM)
  - Daily refresh at midnight UTC
  - Manual refresh available
- **Responsive Design**: Works on desktop and mobile devices
- **No Backend Required**: Pure client-side JavaScript implementation

## File Structure

```
├── index.html          # Main webpage
├── styles.css          # Dark theme styling
├── script.js           # JavaScript functionality
└── README.md           # This documentation
```

## Browser Compatibility

- Modern browsers with ES6+ support
- Chrome, Firefox, Safari, Edge (recent versions)
- JavaScript must be enabled
- Internet connection required for GitHub API access

## Customization

The dashboard can be easily customized by modifying:
- `REPO_OWNER` and `REPO_NAME` in script.js for different repositories
- CSS variables in styles.css for theme colors
- Update intervals in the `setAutoRefresh()` function

## Notes

- Uses GitHub's public API (no authentication required)
- Rate limited to 60 requests per hour for unauthenticated requests
- Best viewed with an internet connection for real-time updates
- Automatically handles API errors and displays appropriate messages
