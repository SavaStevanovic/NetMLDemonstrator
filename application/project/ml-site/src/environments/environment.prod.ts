export const environment = {
  production: true,
  domains: {
    vision: {
      get_filters: 'menager/get_filters',
      frame_upload_stream: 'menager/frame_upload_stream'
    },
    reinforcement: {
      get_filters: 'player/get_filters',
      frame_upload_stream: 'player/frame_upload_stream'
    }
  }
};
