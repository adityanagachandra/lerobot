// Repository configuration
const REPO_OWNER = 'adityanagachandra';
const REPO_NAME = 'lerobot';
const GITHUB_API_BASE = 'https://api.github.com';

// State management
let currentFilter = 'all';
let repositoryData = {
    commits: [],
    files: [],
    issues: [],
    pullRequests: []
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadRepositoryData();
    setAutoRefresh();
});

// Event listeners
function initializeEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', function() {
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
        loadRepositoryData();
    });

    // Activity filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            currentFilter = this.dataset.filter;
            updateActivityDisplay();
        });
    });

    // Modal close
    document.querySelector('.close').addEventListener('click', function() {
        document.getElementById('errorModal').style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        const modal = document.getElementById('errorModal');
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Main data loading function
async function loadRepositoryData() {
    try {
        showLoadingStates();
        
        // Fetch data in parallel
        const [commits, files, issues, pullRequests] = await Promise.all([
            fetchRecentCommits(),
            fetchRecentFiles(),
            fetchActiveIssues(),
            fetchRecentPullRequests()
        ]);

        repositoryData = { commits, files, issues, pullRequests };
        
        updateUI();
        hideLoadingStates();
        
        // Reset refresh button
        const refreshBtn = document.getElementById('refreshBtn');
        refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
        
        console.log('Repository data loaded successfully');
        
    } catch (error) {
        console.error('Error loading repository data:', error);
        showError('Failed to load repository data. Please check your internet connection and try again.');
        hideLoadingStates();
        
        // Reset refresh button
        const refreshBtn = document.getElementById('refreshBtn');
        refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
    }
}

// Fetch recent commits (last 24 hours)
async function fetchRecentCommits() {
    const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const response = await fetch(`${GITHUB_API_BASE}/repos/${REPO_OWNER}/${REPO_NAME}/commits?since=${since}&per_page=20`);
    
    if (!response.ok) throw new Error(`Failed to fetch commits: ${response.status}`);
    
    const commits = await response.json();
    return commits.map(commit => ({
        sha: commit.sha,
        message: commit.commit.message,
        author: commit.commit.author.name,
        date: commit.commit.author.date,
        url: commit.html_url,
        files: commit.files || []
    }));
}

// Fetch recent files (from recent commits)
async function fetchRecentFiles() {
    const commits = await fetchRecentCommits();
    const allFiles = [];
    
    for (const commit of commits.slice(0, 5)) { // Check last 5 commits for files
        try {
            const response = await fetch(`${GITHUB_API_BASE}/repos/${REPO_OWNER}/${REPO_NAME}/commits/${commit.sha}`);
            if (response.ok) {
                const detailedCommit = await response.json();
                if (detailedCommit.files) {
                    detailedCommit.files.forEach(file => {
                        if (file.status === 'added' || file.status === 'modified') {
                            allFiles.push({
                                name: file.filename.split('/').pop(),
                                path: file.filename,
                                status: file.status,
                                additions: file.additions || 0,
                                deletions: file.deletions || 0,
                                changes: file.changes || 0,
                                commitSha: commit.sha,
                                commitDate: commit.date,
                                description: inferFileDescription(file.filename)
                            });
                        }
                    });
                }
            }
        } catch (error) {
            console.warn(`Failed to fetch commit details for ${commit.sha}:`, error);
        }
    }
    
    // Remove duplicates and return latest files
    const uniqueFiles = allFiles.filter((file, index, self) => 
        index === self.findIndex(f => f.path === file.path)
    );
    
    return uniqueFiles.slice(0, 10);
}

// Fetch active issues
async function fetchActiveIssues() {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${REPO_OWNER}/${REPO_NAME}/issues?state=open&per_page=10`);
    
    if (!response.ok) throw new Error(`Failed to fetch issues: ${response.status}`);
    
    const issues = await response.json();
    return issues.filter(issue => !issue.pull_request).map(issue => ({
        number: issue.number,
        title: issue.title,
        body: issue.body,
        state: issue.state,
        labels: issue.labels,
        assignees: issue.assignees,
        created_at: issue.created_at,
        updated_at: issue.updated_at,
        url: issue.html_url,
        user: issue.user
    }));
}

// Fetch recent pull requests
async function fetchRecentPullRequests() {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${REPO_OWNER}/${REPO_NAME}/pulls?state=all&per_page=10&sort=updated`);
    
    if (!response.ok) throw new Error(`Failed to fetch pull requests: ${response.status}`);
    
    const pullRequests = await response.json();
    return pullRequests.map(pr => ({
        number: pr.number,
        title: pr.title,
        body: pr.body,
        state: pr.state,
        created_at: pr.created_at,
        updated_at: pr.updated_at,
        url: pr.html_url,
        user: pr.user,
        merged_at: pr.merged_at
    }));
}

// Update the UI with fetched data
function updateUI() {
    updateStats();
    updateNewFiles();
    updateCurrentWork();
    updateActivityDisplay();
    updateLastUpdateTime();
}

// Update statistics cards
function updateStats() {
    const today = new Date().toDateString();
    const todayCommits = repositoryData.commits.filter(commit => 
        new Date(commit.date).toDateString() === today
    );
    const todayFiles = repositoryData.files.filter(file => 
        new Date(file.commitDate).toDateString() === today
    );

    document.getElementById('newFilesCount').textContent = todayFiles.length;
    document.getElementById('commitsCount').textContent = todayCommits.length;
    document.getElementById('activeIssues').textContent = repositoryData.issues.length;
}

// Update new files section
function updateNewFiles() {
    const container = document.getElementById('newFilesContainer');
    const badge = document.getElementById('newFilesBadge');
    
    if (repositoryData.files.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-file"></i>
                <p>No new files found in recent commits</p>
            </div>
        `;
        badge.textContent = '0';
        return;
    }

    badge.textContent = repositoryData.files.length;
    
    container.innerHTML = repositoryData.files.map(file => `
        <div class="file-item fade-in">
            <div class="file-header">
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-path">${file.path}</div>
                </div>
                <div class="file-size">${formatFileSize(file.changes)} changes</div>
            </div>
            <div class="file-description">${file.description}</div>
            <div class="activity-meta">
                <span><i class="fas fa-plus"></i> ${file.additions} additions</span>
                <span><i class="fas fa-minus"></i> ${file.deletions} deletions</span>
                <span><i class="fas fa-clock"></i> ${formatDate(file.commitDate)}</span>
            </div>
        </div>
    `).join('');
}

// Update current work section
function updateCurrentWork() {
    const container = document.getElementById('workingOnContainer');
    const badge = document.getElementById('workingOnBadge');
    
    const activeWork = [
        ...repositoryData.issues.slice(0, 3).map(issue => ({
            type: 'issue',
            title: issue.title,
            description: issue.body ? issue.body.substring(0, 150) + '...' : 'No description available',
            status: 'in-progress',
            url: issue.url,
            date: issue.updated_at
        })),
        ...repositoryData.pullRequests.filter(pr => pr.state === 'open').slice(0, 2).map(pr => ({
            type: 'pr',
            title: pr.title,
            description: pr.body ? pr.body.substring(0, 150) + '...' : 'No description available',
            status: 'in-progress',
            url: pr.url,
            date: pr.updated_at
        }))
    ];

    if (activeWork.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-tasks"></i>
                <p>No active work items found</p>
            </div>
        `;
        badge.textContent = '0';
        return;
    }

    badge.textContent = activeWork.length;
    
    container.innerHTML = activeWork.map(work => `
        <div class="work-item fade-in">
            <div class="work-header">
                <div>
                    <div class="work-title">
                        <i class="fas fa-${work.type === 'issue' ? 'exclamation-circle' : 'code-branch'}"></i>
                        ${work.title}
                    </div>
                </div>
                <div class="work-status ${work.status}">${work.status.replace('-', ' ')}</div>
            </div>
            <div class="work-description">${work.description}</div>
            <div class="activity-meta">
                <span><i class="fas fa-external-link-alt"></i> <a href="${work.url}" target="_blank">View ${work.type === 'issue' ? 'Issue' : 'PR'}</a></span>
                <span><i class="fas fa-clock"></i> Updated ${formatDate(work.date)}</span>
            </div>
        </div>
    `).join('');
}

// Update activity display based on current filter
function updateActivityDisplay() {
    const container = document.getElementById('recentActivityContainer');
    
    let activities = [];
    
    if (currentFilter === 'all' || currentFilter === 'commits') {
        activities.push(...repositoryData.commits.map(commit => ({
            type: 'commit',
            title: commit.message.split('\n')[0],
            description: `by ${commit.author}`,
            date: commit.date,
            url: commit.url,
            icon: 'code-branch'
        })));
    }
    
    if (currentFilter === 'all' || currentFilter === 'files') {
        activities.push(...repositoryData.files.map(file => ({
            type: 'file',
            title: `${file.status === 'added' ? 'Added' : 'Modified'} ${file.name}`,
            description: file.description,
            date: file.commitDate,
            url: `https://github.com/${REPO_OWNER}/${REPO_NAME}/blob/main/${file.path}`,
            icon: 'file'
        })));
    }
    
    if (currentFilter === 'all' || currentFilter === 'issues') {
        activities.push(...repositoryData.issues.map(issue => ({
            type: 'issue',
            title: issue.title,
            description: `Issue #${issue.number} - ${issue.state}`,
            date: issue.updated_at,
            url: issue.url,
            icon: 'exclamation-circle'
        })));
    }
    
    // Sort by date (newest first)
    activities.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    if (activities.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <p>No recent activity found</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = activities.slice(0, 20).map(activity => `
        <div class="activity-item fade-in">
            <div class="activity-icon ${activity.type}">
                <i class="fas fa-${activity.icon}"></i>
            </div>
            <div class="activity-content">
                <div class="activity-title">${activity.title}</div>
                <div class="activity-description">${activity.description}</div>
                <div class="activity-meta">
                    <span><i class="fas fa-external-link-alt"></i> <a href="${activity.url}" target="_blank">View</a></span>
                    <span><i class="fas fa-clock"></i> ${formatDate(activity.date)}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Utility functions
function inferFileDescription(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const name = filename.split('/').pop();
    
    const descriptions = {
        'py': 'Python source file - likely contains core functionality or utilities',
        'js': 'JavaScript file - frontend logic or Node.js functionality',
        'ts': 'TypeScript file - typed JavaScript for better development experience',
        'md': 'Markdown documentation file',
        'yml': 'YAML configuration file',
        'yaml': 'YAML configuration file',
        'json': 'JSON data or configuration file',
        'txt': 'Text file containing data or documentation',
        'sh': 'Shell script for automation or setup',
        'dockerfile': 'Docker container configuration',
        'requirements.txt': 'Python dependencies specification',
        'package.json': 'Node.js project configuration and dependencies',
        'pyproject.toml': 'Python project configuration and build settings'
    };
    
    if (name.toLowerCase().includes('test')) {
        return 'Test file - contains automated tests for code validation';
    }
    
    if (name.toLowerCase().includes('config')) {
        return 'Configuration file - settings and parameters for the application';
    }
    
    return descriptions[ext] || descriptions[name.toLowerCase()] || 'Source file - part of the project codebase';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0';
    const k = 1024;
    const sizes = ['', 'K', 'M', 'G'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 60) {
        return `${minutes}m ago`;
    } else if (hours < 24) {
        return `${hours}h ago`;
    } else if (days < 7) {
        return `${days}d ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function updateLastUpdateTime() {
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
}

function showLoadingStates() {
    document.querySelectorAll('.loading-state').forEach(el => {
        el.style.display = 'flex';
    });
}

function hideLoadingStates() {
    document.querySelectorAll('.loading-state').forEach(el => {
        el.style.display = 'none';
    });
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorModal').style.display = 'block';
}

// Auto-refresh functionality
function setAutoRefresh() {
    // Refresh every hour during active hours (6 AM to 11 PM)
    setInterval(() => {
        const hour = new Date().getHours();
        if (hour >= 6 && hour <= 23) {
            loadRepositoryData();
        }
    }, 60 * 60 * 1000); // 1 hour

    // Daily refresh at midnight
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(now.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    
    const msUntilMidnight = tomorrow.getTime() - now.getTime();
    
    setTimeout(() => {
        loadRepositoryData();
        // Set up daily refresh
        setInterval(loadRepositoryData, 24 * 60 * 60 * 1000);
    }, msUntilMidnight);
}

// Expose functions for debugging
window.lerobotDashboard = {
    loadRepositoryData,
    repositoryData,
    currentFilter
};