import { AfterViewInit, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { StateService } from '../../services/state/state.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';
import { environment } from 'src/environments/environment';
import { MatSnackBar } from '@angular/material/snack-bar';
import { FilterService } from 'src/app/services/filter/filter.service';
import { SockJS } from 'sockjs-client';

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
  sock: SockJS;
  mask = new Image();

  constructor(
    private snackBar: MatSnackBar,
    public frameService: FrameService,
    private filterService: FilterService,
    private stateService: StateService) {
      this.stateService.videoQuality$
        .subscribe(quality => this.quality = quality)
    }

  ngOnInit(): void {
    this.getFilters();
    this.mask.onload = this.drawScreen.bind(this);
  }

  private drawScreen() {
    this.resizeCanvas()
    this.processed_context.drawImage(this.mask, 0, 0, this.processedCanvas.nativeElement.width, this.processedCanvas.nativeElement.height);
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  toggle_play(play: boolean): void {
    if (play){
      if (this.shouldSendReques()){
        this.setup_stream()
        this.processedCanvas.nativeElement.style.visibility = "visible";
      }
    }
  }

  private shouldSendReques(): boolean {
    for (let i = 0; i < this.filters.length; i++) {
      if (this.filters[i].selectedModel) {
        return true;
      }
    }
    this.snackBar.open('Please select playground environment.', 'Confirm', {
      duration: 2000
    });
    return false;
  }

  private capture():void {
    if (this.sock?.readyState == WebSocket.OPEN){
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
    this.stateService.setVideoPlaying(true)
    this.mask.src = data
  }

  private setupConnection(): void {
    this.sock = this.frameService.openImageConnection(environment.domains.reinforcement.frame_upload_stream);
    this.sock.onmessage = (v) => {
      let data = JSON.parse(v['data'])
      this.processResponse(data);
      requestAnimationFrame(this.capture.bind(this));
    };
  }

  ngAfterViewInit(): void {
    this.processed_context = this.processedCanvas.nativeElement.getContext("2d");
    this.stateService.videoStart$.subscribe(playing => {if (this.filterService.domainSubject.value=="reinforcement") this.toggle_play(playing)})
    this.stateService.menuOpened$.subscribe(opened => this.resizeCanvas())
  }

  resizeCanvas(): void {
    var w = this.processedCanvas.nativeElement.parentElement.clientWidth-20;
    var h = this.processedCanvas.nativeElement.parentElement.clientHeight-20;
    var aspectRatio = 1
    if (this.mask.src)
      aspectRatio = this.mask.width / this.mask.height
    if (w/h > aspectRatio){
      this.processedCanvas.nativeElement.width = h * aspectRatio;
      this.processedCanvas.nativeElement.height = h;
    } else{
      this.processedCanvas.nativeElement.width = w;
      this.processedCanvas.nativeElement.height = w / aspectRatio;
    }
  }
}
