package tensor

import "fmt"

func Fused0066GeluSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	return r, nil
}
func Fused0066GeluSubName() string { return "gelu_sub" }
