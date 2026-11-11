package tensor

import "fmt"

func Fused0011TanhSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	return r, nil
}
func Fused0011TanhSubName() string { return "tanh_sub" }
