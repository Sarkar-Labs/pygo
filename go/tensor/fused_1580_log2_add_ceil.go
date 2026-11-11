package tensor

import "fmt"

func Fused1580Log2AddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Ceil()
	return r, nil
}
func Fused1580Log2AddCeilName() string { return "log2_add_ceil" }
