package tensor

import "fmt"

func Fused0031LogSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	return r, nil
}
func Fused0031LogSubName() string { return "log_sub" }
