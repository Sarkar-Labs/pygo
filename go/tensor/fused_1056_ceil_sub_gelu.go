package tensor

import "fmt"

func Fused1056CeilSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.GELU()
	return r, nil
}
func Fused1056CeilSubGeluName() string { return "ceil_sub_gelu" }
