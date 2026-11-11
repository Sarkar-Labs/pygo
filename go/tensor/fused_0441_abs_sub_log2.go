package tensor

import "fmt"

func Fused0441AbsSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Log2()
	return r, nil
}
func Fused0441AbsSubLog2Name() string { return "abs_sub_log2" }
