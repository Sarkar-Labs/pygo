package tensor

import "fmt"

func Fused0051CeilSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	return r, nil
}
func Fused0051CeilSubName() string { return "ceil_sub" }
