package tensor

import "fmt"

func Fused0440AbsAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Log2()
	return r, nil
}
func Fused0440AbsAddLog2Name() string { return "abs_add_log2" }
