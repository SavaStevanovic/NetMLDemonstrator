import { AfterViewInit, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { StateService } from '../../services/state/state.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';
import { environment } from 'src/environments/environment';
import { FilterService } from 'src/app/services/filter/filter.service';

@Component({
  selector: 'app-playground',
  templateUrl: './playground.component.html',
  styleUrls: ['./playground.component.css']
})
export class PlaygroundComponent implements AfterViewInit, OnInit {
  @ViewChild('processedCanvas') processedCanvas: ElementRef;

  filters: Filter[];
  quality: number;
  processed_context: any;
  sock: any;

  constructor(
    public frameService: FrameService,
    private filterService: FilterService,
    private stateService: StateService) {
      this.stateService.videoQuality$
        .subscribe(quality => this.quality = quality)
    }

  ngOnInit(): void {
    this.getFilters();
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  toggle_play(play: boolean): void {
    if (play){
        this.setup_stream()
    } else {
      if (this.sock) {
        this.sock.close()
      }
      this.setPlaying();
    }
  }

  private shouldSendReques(): boolean {
    if (this.quality==0){
      return false;
    }
    for (let i = 0; i < this.filters.length; i++) {
      if (this.filters[i].selectedModel) {
        return true;
      }
    }
    return false;
  }

  private capture():void {
    this.setPlaying();
    if (!this.stateService.videoPlaying$) {
      this.toggle_play(false)
      return;
    }

    if (this.shouldSendReques()){
      let post_data = {'config': this.filters}
      this.sock.send(JSON.stringify(post_data))
    }
  }

  start_stream(): void {
    this.capture();
  };

  private setup_stream(): void {
    let is_opened_conn = this.sock?.readyState == WebSocket.OPEN;

    if(is_opened_conn) {
      this.start_stream();
    }

    if (!is_opened_conn) {
      this.setupConnection();
      this.sock.onopen = this.start_stream.bind(this)
    }
  }

  private processResponse(data: any): void {
    if (data){
      this.processed_context.drawImage(data, 0, 0, this.processedCanvas.nativeElement.width, this.processedCanvas.nativeElement.height);
    }
  }

  private setupConnection(): void {
    this.sock = this.frameService.openImageConnection(environment.domains.reinforcement.frame_upload_stream);
    this.sock.onmessage = (v) => {
      let data = JSON.parse(v['data'])
      this.processResponse(data);
      requestAnimationFrame(this.capture.bind(this));
    };
  }

  setPlaying(): void {
    this.stateService.setVideoPlaying(this.sock?.readyState == WebSocket.OPEN)
  }

  ngAfterViewInit(): void {
    this.processed_context = this.processedCanvas.nativeElement.getContext("2d");
    this.stateService.videoStart$.subscribe(playing => this.toggle_play(playing))
  }

  resizeCanvas(): void {
    var w = this.processedCanvas.nativeElement.parentElement.clientWidth-20;
    var h = this.processedCanvas.nativeElement.parentElement.clientHeight-20;
    var aspectRatio = 1//this.video_native_element.videoWidth/this.video_native_element.videoHeight
    if (w/h > aspectRatio){
      this.processedCanvas.nativeElement.width = h * aspectRatio;
      this.processedCanvas.nativeElement.height = h;
    } else{
      this.processedCanvas.nativeElement.width = w;
      this.processedCanvas.nativeElement.height = w / aspectRatio;
    }
  }
}
