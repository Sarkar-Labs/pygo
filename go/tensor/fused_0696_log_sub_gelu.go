package tensor

import "fmt"

func Fused0696LogSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.GELU()
	return r, nil
}
func Fused0696LogSubGeluName() string { return "log_sub_gelu" }
