export const formatTime = (time) => {
  if (time === null || Number.isNaN(time)) {
    return "—";
  }
  const totalMs = Math.max(0, Math.round(time * 1000));
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const seconds = Math.floor((totalMs % 60000) / 1000);
  const ms = Math.floor(totalMs % 1000)
    .toString()
    .padStart(3, "0");
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}.${ms}`;
  }
  return `${minutes}:${seconds.toString().padStart(2, "0")}.${ms}`;
};

export const toHex = (buffer) =>
  Array.from(new Uint8Array(buffer))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
