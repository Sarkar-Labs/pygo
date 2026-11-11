package tensor

import "fmt"

func Fused0666LogSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Sin()
	return r, nil
}
func Fused0666LogSubSinName() string { return "log_sub_sin" }
