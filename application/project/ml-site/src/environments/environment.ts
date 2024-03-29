// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

export const environment = {
  production: false,
  domains: {
    vision: {
      get_filters: 'http://0.0.0.0:4321/menager/get_filters',
      frame_upload_stream: 'http://0.0.0.0:4321/menager/frame_upload_stream'
    },
    reinforcement: {
      get_filters: 'http://0.0.0.0:4322/player/get_filters',
      frame_upload_stream: 'http://0.0.0.0:4322/player/frame_upload_stream'
    }
  }
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/dist/zone-error';  // Included with Angular CLI.
