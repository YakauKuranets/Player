export class Orchestrator {
  constructor(context) {
    this.context = context;
    this.blueprints = [];
  }

  register(blueprint) {
    this.blueprints.push(blueprint);
  }

  start() {
    this.blueprints.forEach((blueprint) => blueprint.init(this.context));
  }
}
