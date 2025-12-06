import React, { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  LinearProgress,
  Button,
  TextField,
  InputAdornment,
  IconButton,
  Chip,
  Divider,
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Description as DocumentIcon,
  Category as CategoryIcon,
  Timeline as AnalyticsIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(ArcElement, Toollet, Legend);

// Mock data - replace with actual API calls
const fetchDashboardData = async () => {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        documentCount: 1250,
        clusterCount: 8,
        recentDocuments: [
          { id: 1, title: 'Research Paper on Machine Learning', cluster: 'AI', date: '2023-05-15' },
          { id: 2, title: 'Blockchain Technology Review', cluster: 'Blockchain', date: '2023-05-14' },
          { id: 3, title: 'Quantum Computing Advances', cluster: 'Quantum', date: '2023-05-12' },
        ],
        clusterDistribution: {
          labels: ['AI', 'Blockchain', 'Quantum', 'Cybersecurity', 'Cloud', 'IoT', 'Big Data', 'Other'],
          data: [25, 15, 10, 12, 18, 10, 7, 3],
          backgroundColor: [
            '#FF6384',
            '#36A2EB',
            '#FFCE56',
            '#4BC0C0',
            '#9966FF',
            '#FF9F40',
            '#8AC24A',
            '#FF5252',
          ],
        },
      });
    }, 500);
  });
};

function Dashboard() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  
  const { data, isLoading, error, refetch } = useQuery('dashboardData', fetchDashboardData);

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
    }
  };

  if (isLoading) return <LinearProgress />;
  if (error) return <Typography color="error">Error loading dashboard data</Typography>;

  return (
    <Box>
      <Box mb={4}>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Welcome to the Information Retrieval System
        </Typography>
      </Box>

      {/* Search Bar */}
      <Paper component="form" onSubmit={handleSearch} sx={{ p: 2, mb: 4 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              fullWidth
              variant="contained"
              color="primary"
              size="large"
              type="submit"
              disabled={!searchQuery.trim()}
            >
              Search
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Documents
              </Typography>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h4">{data.documentCount.toLocaleString()}</Typography>
                <DocumentIcon color="primary" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Clusters
              </Typography>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h4">{data.clusterCount}</Typography>
                <CategoryIcon color="secondary" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Search Queries (24h)
              </Typography>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h4">248</Typography>
                <SearchIcon color="success" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Status
              </Typography>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h4">Operational</Typography>
                  <Chip label="All Systems Go" color="success" size="small" />
                </Box>
                <AnalyticsIcon color="action" fontSize="large" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Cluster Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Cluster Distribution"
              action={
                <IconButton onClick={refetch}>
                  <RefreshIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Box height={300}>
                <Doughnut
                  data={{
                    labels: data.clusterDistribution.labels,
                    datasets: [
                      {
                        data: data.clusterDistribution.data,
                        backgroundColor: data.clusterDistribution.backgroundColor,
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'right',
                      },
                    },
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Documents */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Recent Documents"
              action={
                <Button
                  color="primary"
                  size="small"
                  onClick={() => navigate('/search')}
                >
                  View All
                </Button>
              }
            />
            <Divider />
            <CardContent>
              <List>
                {data.recentDocuments.map((doc) => (
                  <React.Fragment key={doc.id}>
                    <ListItem
                      button
                      onClick={() => navigate(`/document/${doc.id}`)}
                      sx={{ px: 0 }}
                    >
                      <ListItemIcon>
                        <DocumentIcon color="action" />
                      </ListItemIcon>
                      <ListItemText
                        primary={doc.title}
                        secondary={`Added on ${doc.date}`}
                        primaryTypographyProps={{
                          style: {
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          },
                        }}
                      />
                      <Chip
                        label={doc.cluster}
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    </ListItem>
                    <Divider component="li" />
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
